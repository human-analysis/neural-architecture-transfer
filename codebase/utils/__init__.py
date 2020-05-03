import numpy as np
import os
import sys

try:
    import horovod.torch as hvd
except ImportError:
    print('No horovod in environment')
    import numpy as hvd
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

from codebase.utils.my_modules import *
from codebase.utils.pytorch_utils import *
from codebase.utils.pytorch_modules import *


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


def get_split_list(in_dim, child_num):
    in_dim_list = [in_dim // child_num] * child_num
    for _i in range(in_dim % child_num):
        in_dim_list[_i] += 1
    return in_dim_list


def list_sum(x):
    if len(x) == 1:
        return x[0]
    else:
        return x[0] + list_sum(x[1:])


def list_weighted_sum(x, weights):
    if len(x) == 1:
        return x[0] * weights[0]
    else:
        return x[0] * weights[0] + list_weighted_sum(x[1:], weights[1:])


def list_mean(x):
    return list_sum(x) / len(x)


def list_mul(x):
    if len(x) == 1:
        return x[0]
    else:
        return x[0] * list_mul(x[1:])


def list_join(val_list, sep='\t'):
    return sep.join([
        str(val) for val in val_list
    ])


def list_continuous_index(val_list, index):
    assert index <= len(val_list) - 1
    left_id = int(index)
    right_id = int(math.ceil(index))
    if left_id == right_id:
        return val_list[left_id]
    else:
        return val_list[left_id] * (right_id - index) + val_list[right_id] * (index - left_id)


def subset_mean(val_list, sub_indexes):
    sub_indexes = int2list(sub_indexes, 1)
    return list_mean([val_list[idx] for idx in sub_indexes])


def delta_ij(i, j):
    if i == j:
        return 1
    else:
        return 0


def sub_filter_start_end(kernel_size, sub_kernel_size):
    center = kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    assert end - start == sub_kernel_size
    return start, end


def int2list(val, repeat_time=1):
    if isinstance(val, list) or isinstance(val, np.ndarray):
        return val
    elif isinstance(val, tuple):
        return list(val)
    else:
        return [val for _ in range(repeat_time)]


def download_url(url, model_dir='~/.torch/', overwrite=False):
    target_dir = url.split('/')[-1]
    model_dir = os.path.expanduser(model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_dir = os.path.join(model_dir, target_dir)
    cached_file = model_dir
    if not os.path.exists(cached_file) or overwrite:
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return cached_file


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    # maxk = max(topk)
    maxk = min(max(topk), int(output.size()[1]))  # in case dataset has less than 5 classes
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        if k > maxk:
            k = maxk
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Horovod: average metrics from distributed training.
class DistributedMetric(object):
    
    def __init__(self, name):
        self.name = name
        self.sum = torch.zeros(1)[0]
        self.count = torch.zeros(1)[0]

    def update(self, val, delta_n=1):
        val = val * delta_n
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.count += delta_n

    @property
    def avg(self):
        return self.sum / self.count
    

class DistributedTensor(object):
    
    def __init__(self, name):
        self.name = name
        self.sum = None
        self.count = torch.zeros(1)[0]
        self.synced = False
    
    def update(self, val, delta_n=1):
        val = val * delta_n
        if self.sum is None:
            self.sum = val.detach()
        else:
            self.sum += val.detach()
        self.count += delta_n

    @property
    def avg(self):
        if not self.synced:
            self.sum = hvd.allreduce(self.sum, name=self.name)
            self.synced = True
        return self.sum / self.count
