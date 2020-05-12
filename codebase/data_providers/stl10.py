import math
import torchvision
import torch.utils.data
import torchvision.transforms as transforms

"""
STL-10 can be automatically download, more details are available from https://ai.stanford.edu/~acoates/stl10/
"""


class STL10DataProvider:

    def __init__(self, save_path=None, train_batch_size=96, test_batch_size=256, valid_size=None,
                 n_worker=2, resize_scale=0.08, distort_color=None, image_size=224, num_replicas=None, rank=None):

        norm_mean = [0.44671097, 0.4398105, 0.4066468]
        norm_std = [0.2603405, 0.25657743, 0.27126738]

        valid_transform = transforms.Compose([
            transforms.Resize(int(math.ceil(image_size / 0.875))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

        valid_data = torchvision.datasets.STL10(
            root=save_path, split='test', download=True, transform=valid_transform)

        self.test = torch.utils.data.DataLoader(
            valid_data, batch_size=test_batch_size, shuffle=False,
            pin_memory=True, num_workers=n_worker)
