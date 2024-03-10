import torch
import torch.nn as nn
import os
from transformers.pytorch_transformers import BertTokenizer

class KnowledgeEmbedder(nn.Module):
    def __init__(self, config,
                 checkpoint,
                 max_q_len, max_k_len, max_k_num) -> None:
        """
        Args:
            tokenizer (nn.Module): pre-trained tokenizer
            checkpoint (str): pre-trained checkpoint name
            max_q_len (int): max question length
            max_k_len (int): max knowledge length
            max_k_num (int): max knowledge sentence numbers
        """
        super().__init__()
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(
            config['path']['raw'], 'best'), do_lower_case=True)
        self.max_q_len = max_q_len
        self.max_k_len = max_k_len
        self.max_k_num = max_k_num
        self.max_len = 50

    def forward(self, label2ans, declaration, question, answer):
        """
        Args:
            knowledge (list): list of topk knowledge
            question (list): list of question words (same size with batch)
        """

        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        for declara, words_q, ans in zip(declaration, question, answer):
            tokens_a = self.tokenizer.tokenize(words_q)
            if len(tokens_a) > 63:
                tokens_a = tokens_a[:(63)]
            tokens = tokens_a + [self.tokenizer.sep_token]
            segment_ids = [0] * len(tokens)
            tokens = [self.tokenizer.cls_token] + tokens
            segment_ids = [0] + segment_ids
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            padding_length = 65 - len(input_ids)
            input_ids = input_ids + ([0] * padding_length)
            input_mask = input_mask + ([0] * padding_length)
            segment_ids = segment_ids + ([0] * padding_length)
            input_mask = input_mask + [1] * 50
            input_mask = input_mask + [1] * self.config['train']['k_max_num']
            input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
            attention_mask_list.append(torch.tensor(input_mask, dtype=torch.long))
            token_type_ids_list.append(torch.tensor(segment_ids, dtype=torch.long))

        inputs_text = { 'input_ids': torch.stack(input_ids_list, dim=0).cuda(),
                        'attention_mask': torch.stack(attention_mask_list, dim=0).cuda(),
                        'token_type_ids': torch.stack(token_type_ids_list, dim=0).cuda(),
                            }
        return inputs_text
