import json
import torch
from tqdm import tqdm
from utils.metrics import soft_acc, top_acc
from param import args

def saved_for_eval(dataloader, results, question_ids, answer_preds, answer):
    """ Save as a format accepted by the evaluation server. """
    scores = soft_acc(answer_preds, answer)
    _, answer_ids = answer_preds.max(dim=1)
    answers = [dataloader.dataset.label2ans[i] for i in answer_ids]
    for q, a, s in zip(question_ids, answers, scores):
        entry = {
            'question_id': q.item(),
            'answer': a,
            'score': s.item(),
        }
        results.append(entry)
    return results


def train(dataset,
        model, optim, lr_scheduler,
        train_loader,
        knowledge_full, knowledge_embed, label2ans,
        tracker, tb_count, writer, ckp):
    loader = tqdm(train_loader, ncols=0)
    loss_trk = tracker.track('loss', tracker.MovingMeanMonitor(momentum=0.99))
    acc_trk = tracker.track('acc', tracker.MovingMeanMonitor(momentum=0.99))
    mlm_dict = {}
    for train_tuple in loader:
        question_words = train_tuple['question']['words']
        inputs_question = {k: v.cuda() for k, v in train_tuple['question']['inputs'].items()}
        answer = train_tuple['answer']['target'].cuda()
        declaration = train_tuple['answer']['declaration']
        question_id = train_tuple['question']['id']
        answer_name_value = train_tuple['answer']['answer_name_value']

        mlm_dict_indice = train_tuple['mlm']['mlm_dict_indice']
        mlm_dict_mask_index = train_tuple['mlm']['mlm_dict_mask_index']
        mlm_dict_top = train_tuple['mlm']['mlm_dict_top']
        img = {
            'id': train_tuple['img']['id'],
            'feat': train_tuple['img']['feat'].cuda(),
            'pos': train_tuple['img']['pos'].cuda() if 'pos' in train_tuple['img'] else None
        }

        loss, vlk_pred, mlm_dict, labels = model(img, #
            question_words, question_id, inputs_question,
            answer, declaration, label2ans,
            knowledge_full, knowledge_embed, answer_name_value, mlm_dict, mlm_dict_indice, mlm_dict_mask_index, mlm_dict_top)

        # define loss functions
        tb_count += 1

        # writer.add_scalar('loss/retriever', retriever_loss.item(), tb_count)
        # writer.add_scalar('loss/reader', reader_loss.item(), tb_count)
        
        loss.backward()
        optim.step()
        optim.zero_grad()

        if dataset == 'OKVQA':
            batch_score = soft_acc(vlk_pred, labels)
        else:
            batch_score = top_acc(vlk_pred, labels)[0]

        fmt = '{:.4f}'.format
        loss_trk.append(loss.item())
        acc_trk.append(batch_score.mean())
        loader.set_postfix(loss=fmt(loss_trk.mean.value),
                            acc=fmt(acc_trk.mean.value))
    if args.do_generate:
        with open('./temp/okvqa/cache/mlm_train_dict.json', 'w') as f:
            json.dump(mlm_dict, f)
    ckp.write_log("acc={:.4f}, loss={:.4f}".format(acc_trk.mean.value, loss_trk.mean.value))
    if lr_scheduler is not None:
            lr_scheduler.step()

    return tb_count


def evaluate(dataset,
        model, dataloader,
        knowledge_full, knowledge_embed, label2ans, write=False):
    score = {'acc': 0.0, 'top3_acc': 0.0}
    results = [] # saving for evaluation
    mlm_dict = {}
    for test_tuple in tqdm(dataloader, ncols=0, leave=True):
        with torch.no_grad():
            question_id = test_tuple['question']['id']
            question_words = test_tuple['question']['words']
            inputs_question = {k: v.cuda() for k, v in test_tuple['question']['inputs'].items()}
            answer = test_tuple['answer']['target'].cuda()
            declaration = test_tuple['answer']['declaration']
            answer_name_value = test_tuple['answer']['answer_name_value']
            mlm_dict_indice = test_tuple['mlm']['mlm_dict_indice']
            mlm_dict_mask_index = test_tuple['mlm']['mlm_dict_mask_index']
            mlm_dict_top = test_tuple['mlm']['mlm_dict_top']
            img = {
                'id': test_tuple['img']['id'],
                'feat': test_tuple['img']['feat'].cuda(),
                'pos': test_tuple['img']['pos'].cuda() if 'pos' in test_tuple['img'] else None
            }
       
            loss, vlk_pred, mlm_dict, labels = model(img, #itm_loss,
                question_words, question_id, inputs_question,
                answer, declaration, label2ans,
                knowledge_full, knowledge_embed, answer_name_value, mlm_dict, mlm_dict_indice, mlm_dict_mask_index, mlm_dict_top)

            if dataset == 'OKVQA':
                batch_score = soft_acc(vlk_pred, labels).sum()
                score['acc'] += batch_score
                if write:
                    results = saved_for_eval(dataloader, results, question_id, vlk_pred, labels)
            else:
                top1_acc, top3_acc = top_acc(vlk_pred, labels)
                score['acc'] += top1_acc.sum()
                score['top3_acc'] += top3_acc.sum()

    if args.do_generate:
        with open('./temp/okvqa/cache/mlm_val_dict.json', 'w') as f:
            json.dump(mlm_dict, f)
    score = {k: v / dataloader.dataset.__len__() for k, v in score.items()}        
    return score, results


def train_generate(dataset,
          model, optim, lr_scheduler,
          train_loader,
          knowledge_full, knowledge_embed, label2ans,
          tracker, tb_count, writer, ckp):
    loader = tqdm(train_loader, ncols=0)
    mlm_dict = {}
    for train_tuple in loader:
        question_words = train_tuple['question']['words']
        inputs_question = {k: v.cuda() for k, v in train_tuple['question']['inputs'].items()}
        answer = train_tuple['answer']['target'].cuda()
        declaration = train_tuple['answer']['declaration']
        question_id = train_tuple['question']['id']
        answer_name_value = train_tuple['answer']['answer_name_value']

        mlm_dict_indice = train_tuple['mlm']['mlm_dict_indice']
        mlm_dict_mask_index = train_tuple['mlm']['mlm_dict_mask_index']
        mlm_dict_top = train_tuple['mlm']['mlm_dict_top']

        img = {
            'id': train_tuple['img']['id'],
            'feat': train_tuple['img']['feat'].cuda(),
            'pos': train_tuple['img']['pos'].cuda() if 'pos' in train_tuple['img'] else None
        }

        loss, vlk_pred, mlm_dict, labels = model(img, #
            question_words, question_id, inputs_question,
            answer, declaration, label2ans,
            knowledge_full, knowledge_embed, answer_name_value, mlm_dict, mlm_dict_indice, mlm_dict_mask_index, mlm_dict_top)
    if args.do_generate:
        with open('./temp/okvqa/cache/mlm_train_dict.json', 'w') as f:
            json.dump(mlm_dict, f)



def evaluate_generate(dataset,
             model, dataloader,
             knowledge_full, knowledge_embed, label2ans, write=False):

    mlm_dict = {}
    for test_tuple in tqdm(dataloader, ncols=0, leave=True):
        with torch.no_grad():
            question_id = test_tuple['question']['id']
            question_words = test_tuple['question']['words']
            inputs_question = {k: v.cuda() for k, v in test_tuple['question']['inputs'].items()}
            answer = test_tuple['answer']['target'].cuda()
            declaration = test_tuple['answer']['declaration']
            answer_name_value = test_tuple['answer']['answer_name_value']
            mlm_dict_indice = test_tuple['mlm']['mlm_dict_indice']
            mlm_dict_mask_index = test_tuple['mlm']['mlm_dict_mask_index']
            mlm_dict_top = test_tuple['mlm']['mlm_dict_top']
            img = {
                'id': test_tuple['img']['id'],
                'feat': test_tuple['img']['feat'].cuda(),
                'pos': test_tuple['img']['pos'].cuda() if 'pos' in test_tuple['img'] else None
            }

            loss, vlk_pred, mlm_dict, labels = model(img,  # itm_loss,
                                                                   question_words, question_id, inputs_question,
                                                                   answer, declaration, label2ans,
                                                                   knowledge_full, knowledge_embed, answer_name_value,
                                                                   mlm_dict, mlm_dict_indice, mlm_dict_mask_index, mlm_dict_top)
    if args.do_generate:
        with open('./temp/okvqa/cache/mlm_val_dict.json', 'w') as f:
            json.dump(mlm_dict, f)
