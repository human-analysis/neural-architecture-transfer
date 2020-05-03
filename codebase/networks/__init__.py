from codebase.networks.proxyless_nets import ProxylessNASNets, proxyless_base, MobileNetV2
from codebase.networks.mobilenet_v3 import MobileNetV3, MobileNetV3Large
from codebase.networks.backbone_mobilenet_v3 import build_mobilenetv3_fpn_backbone


def get_net_by_name(name):
    if name == ProxylessNASNets.__name__:
        return ProxylessNASNets
    elif name == MobileNetV3.__name__:
        return MobileNetV3
    else:
        raise ValueError('unrecognized type of network: %s' % name)
