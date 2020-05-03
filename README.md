# Neural Architecture Transfer
![overview](https://www.zhichaolu.com/assets/neural-architecture-transfer/images/overview.jpg)

## Requirements
``` 
Python >= 3.7.x, PyTorch >= 1.4.0, timm >= 0.1.18 
```

#### ImageNet Classification
![imagenet](https://www.zhichaolu.com/assets/neural-architecture-transfer/images/imagenet.png)

``` shell
python evaluator.py --data /path/to/dataset --model subnets/imagenet/NAT-{M1,M2,M3,M4}/net.config
```

#### Architecture Transfer
![transfer](https://www.zhichaolu.com/assets/neural-architecture-transfer/images/dataset.png)

``` shell
python evaluator.py \
  --data /path/to/dataset \
  --dataset {aircraft,cars,cifar10,cifar100,cinic10,dtd,flowers102,food101,pets,stl10} \
  --model subnets/{dataset}/net-img@{xxx}-flops@{xxx}-top1@{xx.x}/net.config
```


## Acknowledgement 
Codes are modified from [OnceForAll](https://github.com/mit-han-lab/once-for-all) and [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) 
