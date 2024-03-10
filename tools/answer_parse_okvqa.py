import os
import sys
import json
import yaml
from tqdm import tqdm
sys.path.append(os.getcwd())

from param import args
import utils.utils as utils


def get_score(occurences: int):
    """ Average over all 10 choose 9 sets. """
    score_soft = occurences * 0.3
    score = score_soft if score_soft < 1.0 else 1.0
    return score


def filter_answers(ans_dset, min_occurence):
    """ Filtering answers whose frequency is less than min_occurence. """
    occurence = {}
    for ans_entry in tqdm(ans_dset,
                        desc='iter answers',
                        unit_scale=True,
                        total=len(ans_dset)):
        answers = ans_entry['answers']
        for entry in answers:
            answer = entry['answer']
            answer = utils.preprocess_answer(answer)
            if answer not in occurence:
                occurence[answer] = set()
            occurence[answer].add(ans_entry['question_id'])
    for answer in list(occurence.keys()):
        if len(occurence[answer]) < min_occurence:
            occurence.pop(answer)

    print('Num of answers that appear >= {} times: {}'.format(
                                min_occurence, len(occurence)))
    return occurence


def create_ans2label(occurence, cache_root):
    """ Map answers to label. """
    label, label2ans, ans2label = 0, [], {}
    for answer in occurence:
        label2ans.append(answer)
        ans2label[answer] = label
        label += 1

    utils.create_dir(cache_root)

    cache_file = os.path.join(cache_root, 'ans2label.json')
    json.dump(ans2label, open(cache_file, 'w'))
    cache_file = os.path.join(cache_root, 'label2ans.json')
    json.dump(label2ans, open(cache_file, 'w'))
    return ans2label


def compute_target(ans_dset, ans2label, name, cache_root):
    """ Augment ans_dset with soft score as label. """
    target = []
    for ans_entry in tqdm(ans_dset,
                        desc='iter answers',
                        unit_scale=True,
                        total=len(ans_dset)):
        answers = ans_entry['answers']
        answer_count = {}
        for entry in answers:
            answer_ = utils.preprocess_answer(entry['answer'])
            answer_count[answer_] = answer_count.get(answer_, 0) + 1

        labels, scores = [], []
        for answer in answer_count:
            if answer not in ans2label:
                continue
            labels.append(ans2label[answer])
            score = get_score(answer_count[answer])
            scores.append(score)

        target.append({
            'question_type': ans_entry['question_type'],
            'question_id': ans_entry['question_id'],
            'image_id': ans_entry['image_id'],
            'labels': labels,
            'scores': scores,
        })

    utils.create_dir(cache_root)
    cache_file = os.path.join(cache_root, name+'_target.json')
    json.dump(target, open(cache_file, 'w'))


if __name__ == '__main__':
    with open(args.cfg) as stream:
        config = yaml.safe_load(stream)
    with open(utils.path_for(config, train=True, answer=True), 'r') as fd:
        train_answers = json.load(fd)['annotations']
    with open(utils.path_for(config, val=True, answer=True), 'r') as fd:
        val_answers = json.load(fd)['annotations']
   
    answers = train_answers + val_answers
    print("filtering answers less than minimum occurrence...")
    occurence = filter_answers(answers, config['run']['min_occurance'])
    print("create answers to integer labels...")
    ans2label = create_ans2label(occurence, config['path']['cache'])

    print("converting target for train and val answers...")
    compute_target(train_answers, ans2label, 'train', config['path']['cache'])
    compute_target(val_answers, ans2label, 'val', config['path']['cache'])
