import torchvision
import torch.utils.data
import torchvision.transforms as transforms


class CIFAR10DataProvider:

    def __init__(self, save_path=None, train_batch_size=96,
                 test_batch_size=256, valid_size=None,
                 n_worker=2, resize_scale=0.08, distort_color=None,
                 image_size=224, num_replicas=None, rank=None):

        norm_mean = [0.49139968, 0.48215827, 0.44653124]
        norm_std = [0.24703233, 0.24348505, 0.26158768]

        valid_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=3),  # BICUBIC interpolation
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

        valid_data = torchvision.datasets.CIFAR10(
            root=save_path, train=False, download=True, transform=valid_transform)

        self.test = torch.utils.data.DataLoader(
            valid_data, batch_size=test_batch_size, shuffle=False,
            pin_memory=True, num_workers=n_worker)


class CIFAR100DataProvider:

    def __init__(self, save_path=None, train_batch_size=96,
                 test_batch_size=256, valid_size=None,
                 n_worker=2, resize_scale=0.08, distort_color=None,
                 image_size=224, num_replicas=None, rank=None):

        norm_mean = [0.49139968, 0.48215827, 0.44653124]
        norm_std = [0.24703233, 0.24348505, 0.26158768]

        valid_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=3),  # BICUBIC interpolation
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

        valid_data = torchvision.datasets.CIFAR100(
            root=save_path, train=False, download=True, transform=valid_transform)

        self.test = torch.utils.data.DataLoader(
            valid_data, batch_size=test_batch_size, shuffle=False,
            pin_memory=True, num_workers=n_worker)


class CINIC10DataProvider:

    def __init__(self, save_path=None, train_batch_size=96,
                 test_batch_size=256, valid_size=None,
                 n_worker=2, resize_scale=0.08, distort_color=None,
                 image_size=224, num_replicas=None, rank=None):

        norm_mean = [0.47889522, 0.47227842, 0.43047404]
        norm_std = [0.24205776, 0.23828046, 0.25874835]

        valid_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=3),  # BICUBIC interpolation
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

        valid_data = torchvision.datasets.ImageFolder(
            save_path + 'test', transform=valid_transform)

        self.test = torch.utils.data.DataLoader(
            valid_data, batch_size=test_batch_size, shuffle=False,
            pin_memory=True, num_workers=n_worker)