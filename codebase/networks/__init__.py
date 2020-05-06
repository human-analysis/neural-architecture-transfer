from codebase.networks.natnet import NATNet


def get_net_by_name(name):
    if name == NATNet.__name__:
        return NATNet
    else:
        raise ValueError('unrecognized type of network: %s' % name)
