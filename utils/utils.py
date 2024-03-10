import os
from vqa_eval.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval


def path_for(config, train=False, val=False, question=False, answer=False):
    assert train + val == 1
    assert question + answer == 1

    split = 'train' if train else 'val'
    fmt = 'OpenEnded_mscoco_{}2014_questions.json' if question else \
        'mscoco_{}2014_annotations.json'
    return os.path.join(config['path']['raw'], fmt.format(split))


def preprocess_answer(answer):
    """ Mimicing the answer pre-processing with evaluation server. """
    dummy_vqa = lambda: None
    dummy_vqa.getQuesIds = lambda: None
    vqa_eval = VQAEval(dummy_vqa, None)

    answer = vqa_eval.processDigitArticle(
            vqa_eval.processPunctuation(answer))
    answer = answer.replace(',', '')
    return answer


def json_keys2int(x):
    return {int(k): v for k, v in x.items()}
    

def load_folder(folder, suffix):
    imgs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(suffix):
            imgs.append(os.path.join(folder, f))
    return imgs

    
def load_imageid(folder):
    images = load_folder(folder, 'jpg')
    img_ids = set()
    for img in images:
        img_id = int(img.split('/')[-1].split('.')[0].split('_')[-1])
        img_ids.add(img_id)
    return img_ids


def find_path(dir, split, img_id):
    """ Find the coco image path based on image id. 
    Args:
        dir (str): image directory on disk
        split (str): 'train', 'val' or 'test'
        img_id (str): image id without 0
    """
    year = 2015 if split == 'test' else 2014
    padded_zeros = '0' * (12 - len(str(img_id)))
    file_name = 'COCO_{}{}_{}{}.jpg'.format(split, year, padded_zeros, img_id)
    path = os.path.join(dir, '{}{}'.format(split, year), file_name)
    assert os.path.exists(path), 'no image id for {}'.format(img_id)
    return path


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def assert_eq(real, expected):
    assert real == expected, '{} (true) vs {} (expected)'.format(real, expected)


class Tracker:
    """ Keep track of results over time, while having access to
        monitors to display information about them.
    """
    def __init__(self):
        self.data = {}

    def track(self, name, *monitors):
        """ Track a set of results with given monitors under some name (e.g. 'val_acc').
            When appending to the returned list storage, use the monitors
            to retrieve useful information.
        """
        l = Tracker.ListStorage(monitors)
        self.data.setdefault(name, []).append(l)
        return l

    def to_dict(self):
        # turn list storages into regular lists
        return {k: list(map(list, v)) for k, v in self.data.items()}


    class ListStorage:
        """ Storage of data points that updates the given monitors """
        def __init__(self, monitors=[]):
            self.data = []
            self.monitors = monitors
            for monitor in self.monitors:
                setattr(self, monitor.name, monitor)

        def append(self, item):
            for monitor in self.monitors:
                monitor.update(item)
            self.data.append(item)

        def __iter__(self):
            return iter(self.data)

    class MeanMonitor:
        """ Take the mean over the given values """
        name = 'mean'

        def __init__(self):
            self.n = 0
            self.total = 0

        def update(self, value):
            self.total += value
            self.n += 1

        @property
        def value(self):
            return self.total / self.n

    class MovingMeanMonitor:
        """ Take an exponentially moving mean over the given values """
        name = 'mean'

        def __init__(self, momentum=0.9):
            self.momentum = momentum
            self.first = True
            self.value = None

        def update(self, value):
            if self.first:
                self.value = value
                self.first = False
            else:
                m = self.momentum
                self.value = m * self.value + (1 - m) * value
