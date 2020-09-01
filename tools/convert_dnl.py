import argparse
from collections import OrderedDict

import torch

conversion_pairs = {
    'backbone.prefix.conv1': 'backbone.stem.0',
    'backbone.prefix.bn1': 'backbone.stem.1',
    'backbone.prefix.conv2': 'backbone.stem.3',
    'backbone.prefix.bn2': 'backbone.stem.4',
    'backbone.prefix.conv3': 'backbone.stem.6',
    'backbone.prefix.bn3': 'backbone.stem.7',
    'backbone.layer': 'backbone.layer',
    'nlm.conva.0': 'decode_head.convs.0.conv',
    'nlm.conva.1.0': 'decode_head.convs.0.bn',
    'nlm.convb.0': 'decode_head.convs.1.conv',
    'nlm.convb.1.0': 'decode_head.convs.1.bn',
    'nlm.ctb.gamma': 'decode_head.dnl_block.gamma.scale',
    'nlm.ctb.conv_key': 'decode_head.dnl_block.phi.conv',
    'nlm.ctb.conv_query': 'decode_head.dnl_block.theta.conv',
    'nlm.ctb.conv_value': 'decode_head.dnl_block.g.conv',
    'nlm.ctb.conv_mask': 'decode_head.dnl_block.conv_mask',
    'nlm.bottleneck.0': 'decode_head.conv_cat.conv',
    'nlm.bottleneck.1.0': 'decode_head.conv_cat.bn',
    'nlm.bottleneck.3': 'decode_head.conv_seg',
    'dsn.0.conv': 'auxiliary_head.convs.0.conv',
    'dsn.0.bn_relu.0': 'auxiliary_head.convs.0.bn',
    'dsn.2': 'auxiliary_head.conv_seg'
}


def convert(src, dst):
    """Convert keys in detectron pretrained ResNet models to pytorch style."""
    # convert to pytorch style
    state_dict = OrderedDict()
    src_dict = torch.load(src)
    src_state_dict = src_dict.get('state_dict', src_dict)
    for k, v in src_state_dict.items():
        converted = False
        for src_name, dst_name in conversion_pairs.items():
            src_name = f'module.{src_name}'
            if k.startswith(src_name):
                # print('{} is converted'.format(k))
                if k.replace(src_name, dst_name) in state_dict:
                    print('{} is duplicate'.format(k))
                state_dict[k.replace(src_name, dst_name)] = v
                converted = True
                break
        if not converted:
            print('{} not converted'.format(k))

    # save checkpoint
    checkpoint = dict()
    checkpoint['state_dict'] = state_dict
    assert len(state_dict) == len(src_state_dict), '{} vs {}'.format(
        len(state_dict), len(src_state_dict))
    checkpoint['meta'] = dict()
    torch.save(checkpoint, dst)


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src detectron model path')
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()
