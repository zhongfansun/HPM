import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='OKVQA',
                        choices=['OKVQA', 'FVQA'])
    parser.add_argument('--cfg', type=str, default='./cfgs/ok-vqa.yaml',
                        help='configuration yaml file')
    parser.add_argument('--loss-fn', type=str, default='Plain',
                        help='chosen loss function')
    parser.add_argument('--name', type=str, default='unifer.pth',
                        help='saved model name')
    parser.add_argument('--task', type=str, default='mlm_itm',
                        help='task name')
    parser.add_argument('--vlmodel', type=str, default='vilt',
                        help='pretrain model name')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='training batch size')
    parser.add_argument('--resume', action='store_true',
                        help='whether resume from checkpoint')
    parser.add_argument('--test-only', action='store_true',
                        help='evaluate on the test set one time')
    parser.add_argument('--do_generate', action='store_true',
                        help='do_generate')
    parser.add_argument("--gpu", type=str, default='0',
                        help='gpu card ID, split by comma')
    return parser.parse_args()


args = parse_args()
