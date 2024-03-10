import os
import json
import h5py
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers.pytorch_transformers import BertTokenizer
import utils.utils as utils
from param import args
from tqdm import tqdm
import numpy


def _create_entry(config, split, question, answer, decla, answer_name_value, mlm_dict_indice, mlm_dict_mask_index, mlm_dict_top):
    """ Create dataset entries for ok-vqa. 
    Args:
        config (dict): default config file
        split (str): 'train' or 'val, do not support 'test' for ok-vqa
        question (dict): raw question dict
        answer (dict): processed answer dict
    """
    utils.assert_eq(question['question_id'], answer['question_id'])
    utils.assert_eq(question['image_id'], answer['image_id'])

    # print(decla)
    entry = {
        'question': question['question'],
        'question_id': question['question_id'],
        'img_id': question['image_id'],
        'answer': answer,
        'declaration': decla,
        'answer_name_value': answer_name_value,
        'mlm_dict_indice': mlm_dict_indice,
        'mlm_dict_mask_index': mlm_dict_mask_index,
        'mlm_dict_top': mlm_dict_top
    }
    return entry


class BaseDataset(Dataset):
    def __init__(self, split: str, config) -> None:
        super(BaseDataset, self).__init__()
        assert split in ['train', 'test']
        self.config = config
        self.entries = []

        # loading answer-label
        with open(os.path.join(
                config['path']['cache'], 'ans2label.json'), 'r') as fd:
            self.ans2label = json.load(fd)
        with open(os.path.join(
                config['path']['cache'], 'label2ans.json'), 'r') as fd:
            self.label2ans = json.load(fd)
        self.num_ans_candidates = len(self.ans2label)

        # for question tokenizer and image extractor
        self.max_q_len = config['train']['q_max_len']

        if args.vlmodel == 'vinvl':
            self.tokenizer = BertTokenizer.from_pretrained(os.path.join(
                config['path']['raw'], 'best'), do_lower_case=True) #checkpoint-2000000
        else:
            self.tokenizer = eval(config['transformer']['tokenizer']
                                  ).from_pretrained(config['transformer']['checkpoint_token'])

        self.vinvl_features = torch.load(os.path.join(config['path']['img_feat'], 'train_img_frcnn_feats.pt'))
        self.vinvl_val_features = torch.load(os.path.join(config['path']['img_feat'], 'val_img_frcnn_feats.pt'))
        self.vinvl_features.update(self.vinvl_val_features)

        self.extractor = transforms.Compose([
            transforms.Resize(384),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def load_image(self, image_id):
        """ Load one image feature. """
        if not hasattr(self, 'img_id2idx'):
            with open(self.config['path']['feat_ids'], 'r') as fd:
                if type(image_id) == int:  # for ok-vqa dataset
                    self.img_id2idx = json.load(fd, object_hook=utils.json_keys2int)
                else:
                    self.img_id2idx = json.load(fd)
        image_id = self.img_id2idx[image_id]
        if not hasattr(self, 'image_feat'):
            self.image_feat = h5py.File(self.config['path']['img_feat'], 'r')
        features = self.image_feat['features'][image_id]
        spatials = self.image_feat['boxes'][image_id]
        return torch.from_numpy(features), torch.from_numpy(spatials)

    def __getitem__(self, index):
        entry = self.entries[index]

        # image id and path
        img_id = entry['img_id']
        answer_name_value = entry['answer_name_value']

        declaration = entry['declaration']
        # question tokenize
        question_id = entry['question_id']
        question = entry['question']

        mlm_dict_indice = entry['mlm_dict_indice']
        mlm_dict_mask_index = entry['mlm_dict_mask_index']
        mlm_dict_top = entry['mlm_dict_top']

        if args.vlmodel == 'vinvl':
            tokens_a = self.tokenizer.tokenize(question)
            tokens_c = self.tokenizer.tokenize("Answer: " + declaration)
            tokens_a = tokens_a + tokens_c
            if len(tokens_a) > 63:
                tokens_a = tokens_a[:(63)]
            tokens = tokens_a + [self.tokenizer.sep_token]
            segment_ids = [0] * len(tokens)
            tokens = [self.tokenizer.cls_token] + tokens
            segment_ids = [0] + segment_ids
            mask_index = 1   #tokens.index("[MASK]")
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            padding_length = 65 - len(input_ids)
            input_ids = input_ids + ([0] * padding_length)
            input_mask = input_mask + ([0] * padding_length)
            segment_ids = segment_ids + ([0] * padding_length)
            assert len(input_ids) == 65
            assert len(input_mask) == 65
            assert len(segment_ids) == 65
        else:
            inputs_question = self.tokenizer(question,
                                             return_tensors='pt',
                                             max_length=self.max_q_len, padding='max_length', truncation=True)
            inputs_question = {k: v[0] for k, v in inputs_question.items()}

        # for different image inputs
        img_feat = self.vinvl_features[int(img_id)]
        if args.vlmodel == 'vinvl':
            if img_feat.shape[0] > 50:
                img_feat = img_feat[0:50, ]
                input_mask = input_mask + [1] * img_feat.shape[0]
            else:
                input_mask = input_mask + [1] * img_feat.shape[0]
                padding_matrix = torch.zeros((50 - img_feat.shape[0], img_feat.shape[1]))
                img_feat = torch.cat((img_feat, padding_matrix), 0)
                input_mask = input_mask + ([0] * padding_matrix.shape[0])

        if args.vlmodel == 'vinvl':
            inputs_question = {'input_ids': torch.tensor(input_ids, dtype=torch.long),
                                'attention_mask': torch.tensor(input_mask, dtype=torch.long),
                                'token_type_ids': torch.tensor(segment_ids, dtype=torch.long),
                                'mask_index': mask_index}

        # answer tensorize
        answer = entry['answer']
        labels = torch.tensor(answer['labels'], dtype=torch.long)
        scores = torch.tensor(answer['scores'], dtype=torch.float32)
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)

        input_dict = {
            'img': {
                'id': img_id},
            'question': {
                'id': question_id,
                'words': question,
                'inputs': inputs_question,
                'length': len(question) if len(question) < self.max_q_len else self.max_q_len},
            'answer': {
                'declaration': declaration,
                'target': target,
                'answer_name_value': answer_name_value},
            'mlm': {
                'mlm_dict_indice': mlm_dict_indice,
                'mlm_dict_mask_index': mlm_dict_mask_index,
                'mlm_dict_top': mlm_dict_top},
        }
        if args.vlmodel == 'vinvl':
            input_dict['img']['feat'] = img_feat

        return input_dict

    def __len__(self):
        return len(self.entries)


class OKVQADataset(BaseDataset):
    def __init__(self, split: str, config, knowledge_full) -> None:
        super(OKVQADataset, self).__init__(split, config)
        if split == 'test':
            split = 'val'  # different for ok-vqa since there are three split names

        # loading questions
        train = True if split == 'train' else False
        val = True if split == 'val' else False
        question_path = utils.path_for(config, train=train, val=val, question=True)
        with open(question_path, 'r') as fd:
            questions = json.load(fd)['questions']
        questions = sorted(questions, key=lambda x: x['question_id'])

        # loading answers
        with open(os.path.join(config['path']['cache'],
                               '{}_target.json'.format(split)), 'r') as fd:
            answers = json.load(fd)
        answers = sorted(answers, key=lambda x: x['question_id'])
        utils.assert_eq(len(questions), len(answers))
        #########
        with open(os.path.join(config['path']['cache'],
                               'label2ans.json'), 'r') as fd:
            label2ans = json.load(fd)
        with open(os.path.join(config['path']['cache'],
                               'okvqa_final_{}_mavexdecla.json'.format(split)), 'r') as fd:
            declas = json.load(fd)

        with open(os.path.join(config['path']['cache'],
                               'mlm_{}_dict.json'.format(split)), 'r') as fd:
            mlm_dicts = json.load(fd)

        with open(os.path.join(config['path']['cache'],
                               'contain_{}_anwser.json'.format(split)), 'r') as fd:
            contain_anwser_dic = json.load(fd)

        for question, answer in zip(questions, answers):
            answer_name_value = contain_anwser_dic[str(question['question_id'])]
            answer_name_value = answer_name_value.strip('[').strip(']')

            decla = declas[str(question['question_id'])]
            decla = decla.split('<-->')[0]
            mlm_dict_indice = mlm_dicts[str(question['question_id'])]['indice']
            mlm_dict_mask_index = mlm_dicts[str(question['question_id'])]['mask_index']
            mlm_dict_top = mlm_dicts[str(question['question_id'])]['mlm_top']
            self.entries.append(_create_entry(config, split, question, answer, decla, answer_name_value, mlm_dict_indice, mlm_dict_mask_index, mlm_dict_top))


class KnowledgeDataset(Dataset):
    def __init__(self, knowledge_list, max_len, tokenizer) -> None:
        """ for tokenizing knowledge with roberta, supporting both datasets.
        Args:
            knowledge_list (list): list of knowledge sentences
            max_len (int): maximum length of each knowledge sentence
            tokenizer (nn.Module): transformer tokenizer
        """
        super().__init__()
        self.knowledge_list = knowledge_list
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.knowledge_list)

    def __getitem__(self, idx):
        sent = self.knowledge_list[idx]

        tokens_a = self.tokenizer.tokenize(sent)
        if len(tokens_a) > 48:
            tokens_a = tokens_a[:(48)]
        tokens = tokens_a + [self.tokenizer.sep_token]
        segment_ids = [0] * len(tokens)
        tokens = [self.tokenizer.cls_token] + tokens
        segment_ids = [0] + segment_ids
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding_length = 50 - len(input_ids)
        input_ids = input_ids + ([0] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        segment_ids = segment_ids + ([0] * padding_length)

        inputs = {'input_ids': torch.tensor(input_ids, dtype=torch.long),
                            'attention_mask': torch.tensor(input_mask, dtype=torch.long),
                            'token_type_ids': torch.tensor(segment_ids, dtype=torch.long)}

        return inputs
