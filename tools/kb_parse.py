import os
import re
import sys
import json
import gzip
import yaml
import string
from tqdm import tqdm
from collections import Counter
sys.path.append(os.getcwd())
import torch
from torch.utils.data import DataLoader
from transformers.pytorch_transformers import BertConfig
from transformers.pytorch_transformers import BertTokenizer
from modules.model_build import ImageBertForSequenceClassificationPrompt
# logging.set_verbosity_warning()
# logging.set_verbosity_error()

from param import args
from utils.dataset import KnowledgeDataset


def rm_punct(sent):
    """ Remove punctuation in a sentence. """
    table = str.maketrans(dict.fromkeys(string.punctuation))
    return sent.translate(table)


def to_sent_uppercase(words: str) -> str:
    words = re.findall('[a-zA-Z][^A-Z]*', words)
    words = [word.lower() for word in words]
    if len(words):
        return ' '.join(words)


def to_sent_underline(words: str) -> str:
    words = words.split('_')
    words = [word.lower() for word in words]
    if len(words):
        return ' '.join(words)


def filter_rare(triplets: list, entities: list, occurance):
    """ Filtering rare entities. """
    entity_count = Counter(entities)
    entities = [e for e in entity_count if entity_count[e] > occurance]

    # we concate the triplets to a sentence
    knowledge = []
    for triplet in tqdm(triplets, 
                        desc='iter triplets', 
                        unit_scale=True,
                        total=len(triplets)):
        if triplet['head'] in entities and triplet['tail'] in entities:
            if triplet['relation'] is None:
                triplet['relation'] = ''
            knowledge.append(
                to_sent_underline(triplet['head']) + ' ' + \
                to_sent_uppercase(triplet['relation']) + ' ' + \
                to_sent_underline(triplet['tail'])
                )
    return knowledge, entities


@torch.no_grad()
def knowledge_embeddings(config, knowledge, max_len, batch_size=16):
    """ Performing knowledge embedding with Roberta. """

    tokenizer = BertTokenizer.from_pretrained(os.path.join(
        config['path']['raw'], 'best'), do_lower_case=True)  # checkpoint-2000000

    model_name_or_path = os.path.join(config['path']['raw'], 'best')  # checkpoint-2000000
    model_config = BertConfig.from_pretrained(model_name_or_path,
                                              num_labels=3065, finetuning_task='vqa_text',
                                              )
    model_config.img_feature_dim = 2054
    model_config.img_feature_type = 'faster_r-cnn'
    model_config.code_voc = 512
    model_config.hidden_dropout_prob = 0.3
    model_config.loss_type = 'bce'
    model_config.classifier = 'linear'
    model_config.cls_hidden_scale = 3
    model = ImageBertForSequenceClassificationPrompt.from_pretrained(model_name_or_path,
                                                                         from_tf=bool('.ckpt' in model_name_or_path),
                                                                         config=model_config).cuda()

    dataset = KnowledgeDataset(knowledge, max_len, tokenizer)
    loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=0)

    outputs = []
    for inputs in tqdm(loader, desc='iter knowledge',
                    unit_scale=True,
                    total=len(loader)):
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs.extend(model(**inputs)[1].cpu())
    return torch.stack(outputs, dim=0)


def main():
    with open(args.cfg) as stream:
        config = yaml.safe_load(stream)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    triplets = [] # extract knowledge as a list
    relations, entities = [], []

    if args.dataset == 'OKVQA':
        with gzip.open(config['path']['kb'], 'rb') as fd:
            for line in tqdm(fd, 
                            desc='iter raw knowledge', 
                            unit_scale=True,
                            total=len(gzip.open(config['path']['kb'], 'rb').readlines())):
                words = line.decode('utf-8').split('\t')
                if words[2].split('/')[2] == words[3].split('/')[2] == 'en': # english only
                    head = words[2].split('/')[3]
                    relation = words[1].split('/')[-1]
                    tail = words[3].split('/')[-1]

                    relations.append(relation)
                    entities.append(head)
                    entities.append(tail)
                    triplets.append({
                        'head': head,
                        'relation': relation,
                        'tail': tail,
                    })
        knowledge, entities = filter_rare(triplets, entities, occurance=10)
    if args.dataset == 'FVQA':
        with open(config['path']['kb'], 'r') as fd:
            facts = json.load(fd)
        knowledge = [] # extract knowledge as a list
        for fact in tqdm(facts.values(), desc='iter raw facts',
                        unit_scale=True,
                        total=len(facts)):
            knowledge.append(rm_punct(fact['surface']))

    knowledge_embed = knowledge_embeddings(config, knowledge, config['train']['k_max_len'])
    with open(config['path']['knowledge'], 'w') as fd:
        json.dump(knowledge, fd)
    torch.save(knowledge_embed, config['path']['knowledge_embed'])
    print('Number of entries, relations and entities is {}, {}, {}'.format(
        len(knowledge), len(set(relations)), len(set(entities))))
 

if __name__ == '__main__':
    main()
