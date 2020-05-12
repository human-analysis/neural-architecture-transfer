import os
import math

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

"""
The data is available from https://www.robots.ox.ac.uk/~vgg/data/pets/
"""


class OxfordIIITPetsDataProvider:

    def __init__(self, save_path=None, train_batch_size=32, test_batch_size=200, valid_size=None, n_worker=32,
                 resize_scale=0.08, distort_color=None, image_size=224,
                 num_replicas=None, rank=None):

        norm_mean = [0.4828895122298728, 0.4448394893850807, 0.39566558230789783]
        norm_std = [0.25925664613996574, 0.2532760018681693, 0.25981017205097917]

        valid_transform = transforms.Compose([
            transforms.Resize(int(math.ceil(image_size / 0.875))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

        valid_data = datasets.ImageFolder(os.path.join(save_path, 'valid'), valid_transform)

        self.test = torch.utils.data.DataLoader(
            valid_data, batch_size=test_batch_size, shuffle=False,
            pin_memory=True, num_workers=n_worker)