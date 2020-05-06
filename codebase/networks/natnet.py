import shutil

import torch.utils.model_zoo as model_zoo

from codebase.modules.layers import *
from codebase.utils.pytorch_utils import load_state_dict


def drop_connect(inputs, training=False, drop_connect_rate=0.):
    """Apply drop connect."""
    if not training:
        return inputs

    keep_prob = 1 - drop_connect_rate
    random_tensor = keep_prob + torch.rand(
        (inputs.size()[0], 1, 1, 1), dtype=inputs.dtype, device=inputs.device)
    random_tensor.floor_()  # binarize
    output = inputs.div(keep_prob) * random_tensor
    return output


class MobileInvertedResidualBlock(MyModule):

    def __init__(self, mobile_inverted_conv, shortcut, drop_connect_rate=0.0):
        super(MobileInvertedResidualBlock, self).__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut
        self.drop_connect_rate = drop_connect_rate

    def forward(self, x):
        if self.mobile_inverted_conv is None or isinstance(self.mobile_inverted_conv, ZeroLayer):
            res = x
        elif self.shortcut is None or isinstance(self.shortcut, ZeroLayer):
            res = self.mobile_inverted_conv(x)
        else:
            # res = self.mobile_inverted_conv(x) + self.shortcut(x)
            res = self.mobile_inverted_conv(x)

            if self.drop_connect_rate > 0.:
                res = drop_connect(res, self.training, self.drop_connect_rate)

            res += self.shortcut(x)

        return res

    @property
    def module_str(self):
        return '(%s, %s)' % (
            self.mobile_inverted_conv.module_str if self.mobile_inverted_conv is not None else None,
            self.shortcut.module_str if self.shortcut is not None else None
        )

    @property
    def config(self):
        return {
            'name': MobileInvertedResidualBlock.__name__,
            'mobile_inverted_conv': self.mobile_inverted_conv.config if self.mobile_inverted_conv is not None else None,
            'shortcut': self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config):
        mobile_inverted_conv = set_layer_from_config(config['mobile_inverted_conv'])
        shortcut = set_layer_from_config(config['shortcut'])
        return MobileInvertedResidualBlock(mobile_inverted_conv, shortcut, drop_connect_rate=config['drop_connect_rate'])


class NATNet(MyNetwork):
    """ variants of MobileNet-v3 """
    def __init__(self, first_conv, blocks, final_expand_layer, feature_mix_layer, classifier):
        super(NATNet, self).__init__()

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.final_expand_layer = final_expand_layer
        self.feature_mix_layer = feature_mix_layer
        self.classifier = classifier

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_expand_layer(x)
        x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling
        x = self.feature_mix_layer(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        for block in self.blocks:
            _str += block.module_str + '\n'
        _str += self.final_expand_layer.module_str + '\n'
        _str += self.feature_mix_layer.module_str + '\n'
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            'name': NATNet.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'final_expand_layer': self.final_expand_layer.config,
            'feature_mix_layer': self.feature_mix_layer.config,
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config, drop_connect_rate=0.0, pretrained=False):
        first_conv = set_layer_from_config(config['first_conv'])
        final_expand_layer = set_layer_from_config(config['final_expand_layer'])
        feature_mix_layer = set_layer_from_config(config['feature_mix_layer'])
        classifier = set_layer_from_config(config['classifier'])

        blocks = []
        for block_idx, block_config in enumerate(config['blocks']):
            block_config['drop_connect_rate'] = drop_connect_rate * block_idx / len(config['blocks'])
            blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))

        net = NATNet(first_conv, blocks, final_expand_layer, feature_mix_layer, classifier)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)

        if pretrained:
            cache_dir = ".tmp"
            state_dict = model_zoo.load_url(config['url'], model_dir=cache_dir, progress=True, map_location='cpu')
            state_dict = load_state_dict(state_dict)
            net.load_state_dict(state_dict)
            shutil.rmtree(cache_dir)

        return net

    def zero_last_gamma(self):
        for m in self.modules():
            if isinstance(m, MobileInvertedResidualBlock):
                if isinstance(m.mobile_inverted_conv, MBInvertedConvLayer) and isinstance(m.shortcut, IdentityLayer):
                    m.mobile_inverted_conv.point_linear.bn.weight.data.zero_()

    @staticmethod
    def build_net_via_cfg(cfg, input_channel, last_channel, n_classes, dropout_rate):
        # first conv layer
        first_conv = ConvLayer(
            3, input_channel, kernel_size=3, stride=2, use_bn=True, act_func='h_swish', ops_order='weight_bn_act'
        )
        # build mobile blocks
        feature_dim = input_channel
        blocks = []
        for stage_id, block_config_list in cfg.items():
            for k, mid_channel, out_channel, use_se, act_func, stride, expand_ratio in block_config_list:
                mb_conv = MBInvertedConvLayer(
                    feature_dim, out_channel, k, stride, expand_ratio, mid_channel, act_func, use_se
                )
                if stride == 1 and out_channel == feature_dim:
                    shortcut = IdentityLayer(out_channel, out_channel)
                else:
                    shortcut = None
                blocks.append(MobileInvertedResidualBlock(mb_conv, shortcut))
                feature_dim = out_channel
        # final expand layer
        final_expand_layer = ConvLayer(
            feature_dim, feature_dim * 6, kernel_size=1, use_bn=True, act_func='h_swish', ops_order='weight_bn_act',
        )
        feature_dim = feature_dim * 6
        # feature mix layer
        feature_mix_layer = ConvLayer(
            feature_dim, last_channel, kernel_size=1, bias=False, use_bn=False, act_func='h_swish',
        )
        # classifier
        classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)

        return first_conv, blocks, final_expand_layer, feature_mix_layer, classifier

    @staticmethod
    def adjust_cfg(cfg, ks=None, expand_ratio=None, depth_param=None, stage_width_list=None):
        for i, (stage_id, block_config_list) in enumerate(cfg.items()):
            for block_config in block_config_list:
                if ks is not None and stage_id != '0':
                    block_config[0] = ks
                if expand_ratio is not None and stage_id != '0':
                    block_config[-1] = expand_ratio
                    block_config[1] = None
                    if stage_width_list is not None:
                        block_config[2] = stage_width_list[i]
            if depth_param is not None and stage_id != '0':
                new_block_config_list = [block_config_list[0]]
                new_block_config_list += [copy.deepcopy(block_config_list[-1]) for _ in range(depth_param - 1)]
                cfg[stage_id] = new_block_config_list
        return cfg

    def reset_classifier(self, last_channel, n_classes, dropout_rate=0.0):
        self.classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)
