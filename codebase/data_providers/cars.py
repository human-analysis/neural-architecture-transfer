import os
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

"""
The data is available from https://ai.stanford.edu/~jkrause/cars/car_dataset.html
use bounding boxes provided to extract cars from background
"""


class StanfordCarsDataProvider:

    def __init__(self, save_path=None, train_batch_size=32, test_batch_size=200, valid_size=None, n_worker=32,
                 resize_scale=0.08, distort_color=None, image_size=224,
                 num_replicas=None, rank=None):

        norm_mean = [0.4105214534294453, 0.38574356611082533, 0.3959628699849632]
        norm_std = [0.28611458811352136, 0.2801378084154138, 0.28520087594365295]

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