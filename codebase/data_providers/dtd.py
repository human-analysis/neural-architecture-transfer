import os
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

"""
The data is available from https://www.robots.ox.ac.uk/~vgg/data/dtd/
"""


class DTDDataProvider:

    def __init__(self, save_path=None, train_batch_size=32, test_batch_size=200, valid_size=None, n_worker=32,
                 resize_scale=0.08, distort_color=None, image_size=224,
                 num_replicas=None, rank=None):

        norm_mean = [0.5329876098715876, 0.474260843249454, 0.42627281899380676]
        norm_std = [0.26549755708788914, 0.25473554309855373, 0.2631728035662832]

        valid_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=3),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

        valid_data = datasets.ImageFolder(os.path.join(save_path, 'valid'), valid_transform)

        self.test = torch.utils.data.DataLoader(
            valid_data, batch_size=test_batch_size, shuffle=False,
            pin_memory=True, num_workers=n_worker)