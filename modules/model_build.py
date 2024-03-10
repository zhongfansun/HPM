import torch
import torch.nn as nn
from transformers.pytorch_transformers import BertConfig
from transformers.pytorch_transformers.modeling_bert import BertPooler
from modules.modal_embed import KnowledgeEmbedder    #, DeclarationEmbedder
from param import args
import numpy
import torch.nn.functional as F
import os
from modeling_bert import ImageBertForSequenceClassification


class ImageBertForSequenceClassificationPrompt(ImageBertForSequenceClassification):
    def __init__(self, config):
        super(ImageBertForSequenceClassificationPrompt, self).__init__(config)

        self.mask_pooler = BertPooler(config)

        if hasattr(config, 'classifier'):
            if not hasattr(config, 'cls_hidden_scale'):
                config.cls_hidden_scale = 2

            if config.classifier == 'linear':
                self.classifier = nn.Linear(config.hidden_size,
                                            3129)
            elif config.classifier == 'mlp':
                self.classifier = nn.Sequential(
                    nn.Linear(config.hidden_size * 2, config.hidden_size * config.cls_hidden_scale),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size * config.cls_hidden_scale, self.config.num_labels)
                )
        else:
            self.classifier = nn.Linear(config.hidden_size * 2, self.config.num_labels)  # original
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
            position_ids=None, head_mask=None, img_feats=None, mask_index=None, topk_embed_knowledge=None):
        # (batch_size, sequence_length, hidden_dim), (batch_size, hidden_dim), ...

        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask, img_feats=img_feats,
                            topk_embed_knowledge=topk_embed_knowledge)

        return outputs[0], outputs[1]

class KnowledgeNotNoise(nn.Module):
    def __init__(self, config, loss_fn, num_answers) -> None:
        """
        Args:
            config (str): default config file
            loss_fn (nn.Module): loss function for vqa, CE or BCE
            num_answers (int): number of candidate answers
        """
        super().__init__()
        self.config = config
        self.loss_fn = loss_fn
        self.num_answers = num_answers
        self.topk = config['train']['k_max_num']
        # self.knowledge_embedding = nn.Embedding.from_pretrained(knowledge_embed, freeze=False)
        # self.knowledge_embedding = nn.Parameter(knowledge_embed,requires_grad=True)
        self.linear_vision = nn.Sequential(nn.Linear(1024, 1024))

        if args.vlmodel == 'vinvl':
            model_name_or_path = os.path.join(config['path']['raw'], 'best')#checkpoint-2000000
            model_config = BertConfig.from_pretrained(model_name_or_path,
                num_labels=num_answers, finetuning_task='vqa_text',
            )
            model_config.img_feature_dim = 2054
            model_config.img_feature_type = 'faster_r-cnn'
            model_config.code_voc = 512
            model_config.hidden_dropout_prob = 0.3
            model_config.loss_type = 'bce'
            model_config.classifier = 'linear'
            model_config.cls_hidden_scale = 3
            model_mlm = ImageBertForSequenceClassificationPrompt.from_pretrained(model_name_or_path,
                                                from_tf=bool('.ckpt' in model_name_or_path), config=model_config)
        else:
            model_mlm = eval(config['transformer']['vlp_model']
                ).from_pretrained(config['transformer']['checkpoint_model'])



        # model_itm = eval(config['transformer']['vlp_model']
        #     ).from_pretrained(config['transformer']['checkpoint_model'])

        self.mlm_model = VLModel(model_mlm, config)
        self.itm_model = VLModel(model_mlm, config)

        self.sim_func = nn.CosineSimilarity(dim=-1)

        # for retriever supervision
        self.reader_loss = loss_fn
        # self.declaration_embedder = DeclarationEmbedder(
        #     eval(config['transformer']['tokenizer']),
        #     config['transformer']['checkpoint_token'],
        #     max_q_len=config['train']['q_max_len'],
        #     max_k_len=config['train']['k_max_len'],
        #     max_k_num=config['train']['k_max_num'],
        # )
        self.sim_func = nn.CosineSimilarity(dim=-1)
        self.retriever = Retriever(config['train']['k_max_num'], self.sim_func)

        hidden_size = config['train']['hidden_size']
        fusion_modules = [nn.Dropout(config['run']['dropout'])]

        # fusion_modules.append(nn.Linear(hidden_size * 2, hidden_size * 2))
        # fusion_modules.append(nn.LayerNorm(hidden_size * 2))
        # fusion_modules.append(nn.GELU())
        # fusion_modules += list(self.vl_model.fusion_mlp.children())

        # fusion_modules.append(nn.Dropout(config['run']['dropout']))
        self.fusion_mlp = nn.Sequential(*fusion_modules)
        if args.vlmodel == 'vinvl':
            hidden_size = 1024
        if config['transformer']['model_name'] == 'visualbert':
            fusion_size = hidden_size
        else:
            fusion_size = hidden_size * 2

        self.itmpredict = nn.Sequential(nn.Dropout(config['run']['dropout']),
                                        nn.Linear(fusion_size, hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size, 1)
                                        )

        self.mlmclassifier = nn.Parameter(torch.Tensor(hidden_size * 2, num_answers))
        self.mlmclassifier.data.uniform_(-1 / hidden_size, 1 / hidden_size)
        self.result = []
        # self.linear_vision = nn.Linear(768*2, 768)

        self.knowledgeitm = KnowledgeITM(config, self.itm_model)

    def l2_norm(self, input, dim=-1):
        norm = torch.norm(input, dim=dim, keepdim=True)
        output = torch.div(input, norm)
        return output

    def mlmpredict(self, feature, scale=16):
        """ mapped into cosine space
        Args:
            feature (tensor): final fused feature
            scale (int, optional): scale up logits (essential). Defaults to 16.
        """
        fusion = self.fusion_mlp(feature)
        fusion = self.l2_norm(fusion, dim=-1)
        kernel = self.l2_norm(self.mlmclassifier, dim=0)
        logits = scale * torch.mm(fusion, kernel)
        return logits

    def forward(self, img, 
            question, question_id, inputs_question,
            answer, declaration, label2ans,
            knowledge_full, knowledge_embed, answer_name_value, mlm_dict, mlm_dict_indice, mlm_dict_mask_index,
            mlm_dict_top):
        """
        Args:
            img (dict): img features
            question (str): raw question words
            inputs_question (dict): question related feature
            answer (tensor): ground-truth answer vectors
            knowledge_full (list): list of knowledge sentences
            knowledge_embed (tensor): same order with knowledge_full
        """
        #vl = self.vl_model(img, inputs_question, 'vl') # (b, hidden_size)768
        batch_size = answer.shape[0]

        vldcls, vld, anchor_mask_itm = self.mlm_model(img, inputs_question, 'mlm')

        top_indice = self.retriever(vldcls, knowledge_embed)

        topk_knowledge, topk_embed = [], []
        for each_top_indice in top_indice:
            topk_embed.append(knowledge_embed[each_top_indice.tolist()])
            topk_knowledge.append([knowledge_full[idx] for idx in each_top_indice.tolist()])
        topk_embed_single = torch.stack(topk_embed, dim=0) #16,5,1024

        topk_embed_single = self.linear_vision(topk_embed_single)


        vl_k = self.knowledgeitm(img, label2ans, question, declaration, anchor_mask_itm, answer, topk_embed_single)
        vl_k_logits = self.mlmpredict(torch.cat([vl_k, vld], 1))
        mlm_loss = self.reader_loss(vl_k_logits, answer)


        fina_logits = vl_k_logits
        fina_labels = answer
        loss = mlm_loss.mean()


        return loss, fina_logits, mlm_dict, fina_labels #,

class Retriever(nn.Module):
    def __init__(self, topk, sim_func) -> None:
        """
        Args:
            topk (int): return topk most relevant knowledge
            sim_func (nn.Module): similarity function, default cosine similarity
        """
        super().__init__()
        self.topk = topk
        self.sim_func = sim_func

    @torch.no_grad()
    def forward(self, query, knowledge_embed): #knowledge_embed：1199797, 768
        """ Retrieve topk knoweldge in non-batch manner for resource reason. """
        top_indices = []
        for query_single in query:
            # knowledge_indexs = torch.zeros(len(knowledge_full)).cuda()  # 19196752
            sims = torch.matmul(query_single.unsqueeze(0), torch.transpose(knowledge_embed, 0, 1))
            top_value, indices = sims.squeeze(0).topk(self.topk, dim=-1) #5 .squeeze(0)
            top_indices.append(indices)

        return top_indices

class KnowledgeITM(nn.Module):
    def __init__(self, config, itm_model) -> None:
        super().__init__()
        self.itm_model = itm_model
        self.topk = config['train']['k_max_num']
        self.knowledge_embedder = KnowledgeEmbedder(config,
            config['transformer']['checkpoint_token'],
            max_q_len=config['train']['q_max_len'],
            max_k_len=config['train']['k_max_len'],
            max_k_num=config['train']['k_max_num'],
        )

    def forward(self, img, label2ans, question, declaration, anchor_mask_pre, answer, topk_embed_expend):
        batch_size = img['feat'].shape[0]
        # incorporating knowledge info
        inputs_text = self.knowledge_embedder(label2ans, declaration, question, answer) # (b * k, )
        # 16,50,1024-->80,50,1024
        vl_k, _, _ = self.itm_model(img, inputs_text, 'itm', anchor_mask_pre, topk_embed_expend)
        #print(vl_k['output1'].shape, vl_k['output2'].shape)

        return vl_k


class VLModel(nn.Module):
    def __init__(self, model, config) -> None:
        """ Build a general VLModel for knowledge-based VQA. 
        Args:
            name: currently support three ('vilt', 'lxmert', 'visualbert')    
        """
        super().__init__()
        self.model = model
        self.name = config['transformer']['model_name']
        self.topk = config['train']['k_max_num']
        if self.name == 'vilt':
            self.fusion_mlp = nn.Sequential(*list(model.classifier.children())[:-1])
        if self.name == 'lxmert':
            self.fusion_mlp = nn.Sequential(*list(model.answer_head.logit_fc.children())[:-1])
        if self.name == 'visualbert':
            self.fusion_mlp = model.dropout # only one dropout layer

        self.multihead_fusion = nn.MultiheadAttention(768, 8, dropout=0.1, batch_first=True)

    def forward(self, img, text, vld_default, anchor_mask_pre=None, topk_embed_expend=None):
        if args.vlmodel == 'vinvl':
            if topk_embed_expend is not None:
                inputs = {'img_feats': img['feat'], 'topk_embed_knowledge': topk_embed_expend}
            else:
                inputs = {'img_feats': img['feat']}
            inputs.update(text)
            pooled_output, pooled_output_cls = self.model(**inputs)

            if vld_default == 'mlm':
                anchor_maskid = torch.cat([text['input_ids'], torch.ones_like(text['input_ids'], dtype=torch.float)], 1)
                anchor_mask = anchor_maskid == 103
                anchor_mask = anchor_mask[:,0:115]
            else:
                anchor_mask = anchor_mask_pre

            pooled_output = pooled_output[:,0:115,:]

            anchor = pooled_output[anchor_mask]

            return pooled_output_cls, anchor, anchor_mask

        if args.vlmodel == 'vilt':
            if topk_embed_expend is not None:
                inputs = {'pixel_values': img['feat'], 'topk_embed_knowledge': topk_embed_expend}
            else:
                inputs = {'pixel_values': img['feat']}
            inputs.update(text)
            outputs = self.model.vilt(**inputs)

            if vld_default == 'mlm':
                anchor_mask = text['input_ids'] == 103
            else:
                batchsize = anchor_mask_pre.shape[0]
                anchor_mask = torch.cat([anchor_mask_pre] * self.topk, -1).view(batchsize * self.topk, -1)

            text_seq_len = text['input_ids'].shape[1]
            text_features, _ = (outputs.last_hidden_state[:, :text_seq_len], outputs.last_hidden_state[:, text_seq_len:])
            anchor = torch.cat([outputs.pooler_output, text_features[anchor_mask]], -1)

            return outputs.pooler_output, anchor, anchor_mask


        if self.name == 'lxmert':
            inputs = {
                'visual_feats': img['feat'],
                'visual_pos': img['pos'],
                'return_dict': True,
                'output_attentions': True
            }
            inputs.update(text)
            # lxmert_output = self.model.lxmert(**inputs)
            # return lxmert_output[2], lxmert_output.cross_encoder_attentions[-1]
            return self.model.lxmert(**inputs)[2]

        if self.name == 'visualbert':
            inputs = {'visual_embeds': img['feat']}
            inputs.update(text)
            return self.model.visual_bert(**inputs)[1]
