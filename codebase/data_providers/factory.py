def get_dataloader(**kwargs):
    if kwargs['dataset'] == 'imagenet':
        from codebase.data_providers.imagenet import ImagenetDataProvider
        loader_class = ImagenetDataProvider
    elif kwargs['dataset'] == 'cifar10':
        from codebase.data_providers.cifar import CIFAR10DataProvider
        loader_class = CIFAR10DataProvider
    elif kwargs['dataset'] == 'cifar100':
        from codebase.data_providers.cifar import CIFAR100DataProvider
        loader_class = CIFAR100DataProvider
    elif kwargs['dataset'] == 'cinic10':
        from codebase.data_providers.cifar import CINIC10DataProvider
        loader_class = CINIC10DataProvider
    elif kwargs['dataset'] == 'aircraft':
        from codebase.data_providers.aircraft import FGVCAircraftDataProvider
        loader_class = FGVCAircraftDataProvider
    elif kwargs['dataset'] == 'cars':
        from codebase.data_providers.cars import StanfordCarsDataProvider
        loader_class = StanfordCarsDataProvider
    elif kwargs['dataset'] == 'dtd':
        from codebase.data_providers.dtd import DTDDataProvider
        loader_class = DTDDataProvider
    elif kwargs['dataset'] == 'flowers102':
        from codebase.data_providers.flowers102 import Flowers102DataProvider
        loader_class = Flowers102DataProvider
    elif kwargs['dataset'] == 'food101':
        from codebase.data_providers.food import Food101DataProvider
        loader_class = Food101DataProvider
    elif kwargs['dataset'] == 'pets':
        from codebase.data_providers.pets import OxfordIIITPetsDataProvider
        loader_class = OxfordIIITPetsDataProvider
    elif kwargs['dataset'] == 'stl10':
        from codebase.data_providers.stl10 import STL10DataProvider
        loader_class = STL10DataProvider
    else:
        raise NotImplementedError

    loader = loader_class(
        save_path=kwargs['data'], test_batch_size=kwargs['test_batch_size'],
        n_worker=kwargs['n_worker'], image_size=kwargs['image_size'])

    return loader
