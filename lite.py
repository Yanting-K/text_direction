import argparse
import os

import paddle
import yaml



class MyNet(paddle.nn.Layer):
    def __init__(self):
        super(MyNet,self).__init__()
        self.resnet = paddle.vision.models.mobilenet_v2(pretrained=True, num_classes=0)
        self.flatten = paddle.nn.Flatten()
        self.linear = paddle.nn.Linear(1280, 4)

    def forward(self, img):
        img = img['img']
        y = self.resnet(img)
        y = self.flatten(y)
        out = self.linear(y)

        return out

def parse_args():
    parser = argparse.ArgumentParser(description='Model export.')
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the exported model',
        type=str,
        default='./output')
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for export',
        type=str,
        default=None)
    parser.add_argument(
        "--input_shape",
        nargs='+',
        help="Export the model with fixed input shape, such as 1 3 1024 1024.",
        type=int,
        default=None)

    return parser.parse_args()


def main(args):
    os.environ['PADDLESEG_EXPORT_STAGE'] = 'True'

    net = MyNet()
    net.eval()
    para_state_dict = paddle.load(args.model_path)
    net.set_dict(para_state_dict)

    if args.input_shape is None:
        shape = [None, 3, None, None]
    else:
        shape = args.input_shape

    input_spec = [{"img": paddle.static.InputSpec(shape=shape, name='img')}]


    net = paddle.jit.to_static(net, input_spec=input_spec)
    save_path = os.path.join(args.save_dir, 'model')
    paddle.jit.save(net, save_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
