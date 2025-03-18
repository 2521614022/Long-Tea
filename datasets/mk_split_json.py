import os
import random
from collections import defaultdict
import json
import os.path as osp


class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath='', label=0, domain=-1, classname=''):
        assert isinstance(impath, str)
        assert isinstance(label, int)
        assert isinstance(domain, int)
        assert isinstance(classname, str)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname


def read_data(path):
    filepath = path
    items = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            imname, label = line.split(' ')
            breed = imname.split('_')[:-1]
            breed = '_'.join(breed)
            breed = breed.lower()
            imname += '.jpg'
            impath = os.path.join("data", imname)
            label = int(label) - 1 # convert to 0-based index
            item = Datum(
                impath=impath,
                label=label,
                classname=breed
            )
            items.append(item)
        
    return items


def split_trainval(trainval, p_val=0.2):
    p_trn = 1 - p_val
    print(f'Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val')
    tracker = defaultdict(list)
    for idx, item in enumerate(trainval):
        label = item.label
        tracker[label].append(idx)
    
    train, val = [], []
    for label, idxs in tracker.items():
        n_val = round(len(idxs) * p_val)
        assert n_val > 0
        random.shuffle(idxs)
        for n, idx in enumerate(idxs):
            item = trainval[idx]
            if n < n_val:
                val.append(item)
            else:
                train.append(item)
    
    return train, val
    
    
def write_json(obj, fpath):
    """Writes to a json file."""
    if not osp.exists(osp.dirname(fpath)):
        os.makedirs(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))    
    
    
def save_split(train, val, test, filepath, path_prefix):
    def _extract(items):
        out = []
        for item in items:
            impath = item.impath
            label = item.label
            classname = item.classname
            impath = impath.replace(path_prefix, '')
            if impath.startswith('\\'):
                impath = impath[1:]
            out.append((impath, label, classname))
        return out
    
    train = _extract(train)
    val = _extract(val)
    test = _extract(test)

    split = {
        'train': train,
        'val': val,
        'test': test
    }

    write_json(split, filepath)
    print(f'Saved split to {filepath}')
    

def main():
    trainval = read_data("trainval.txt")
    test = read_data("test.txt")
    train, val = split_trainval(trainval)
    save_split(train, val, test, './split_tea.json', 'data')


if __name__ == '__main__':
    main()