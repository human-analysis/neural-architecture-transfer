import os
import math

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

"""
The data is available from https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
"""


class Flowers102DataProvider:
    
    def __init__(self, save_path=None, train_batch_size=32, test_batch_size=512, valid_size=None, n_worker=32,
                 resize_scale=0.08, distort_color=None, image_size=224,
                 num_replicas=None, rank=None):

        norm_mean = [0.5178361839861569, 0.4106749456881299, 0.32864167836880803]
        norm_std = [0.2972239085211309, 0.24976049135203868, 0.28533308036347665]

        valid_transform = transforms.Compose([
            transforms.Resize(int(math.ceil(image_size / 0.875))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

        valid_data = datasets.ImageFolder(os.path.join(save_path, 'test'), valid_transform)

        self.test = torch.utils.data.DataLoader(
            valid_data, batch_size=test_batch_size, shuffle=False,
            pin_memory=True, num_workers=n_worker)