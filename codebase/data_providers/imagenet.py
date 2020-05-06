import os
import warnings
from timm.models import create_model
from timm.data import Dataset, create_loader, resolve_data_config


class ImagenetDataProvider:

    def __init__(self, save_path=None, train_batch_size=256, test_batch_size=512,
                 valid_size=None, n_worker=32, resize_scale=0.08, distort_color=None,
                 image_size=224, tf_preprocessing=False, num_replicas=None,
                 rank=None, use_prefetcher=False, pin_memory=False, fp16=False):

        warnings.filterwarnings('ignore')

        dataset = Dataset(os.path.join(save_path, "val"), load_bytes=tf_preprocessing)
        dummy_model = create_model('efficientnet_b0')
        data_config = resolve_data_config({}, model=dummy_model)

        test_loader = create_loader(
            dataset,
            input_size=image_size,
            batch_size=test_batch_size,
            use_prefetcher=use_prefetcher,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=n_worker,
            crop_pct=data_config['crop_pct'],
            pin_memory=pin_memory,
            fp16=fp16,
            tf_preprocessing=None)

        self.test = test_loader