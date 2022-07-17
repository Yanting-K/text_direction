import os
import sys
import glob
import cv2
import paddle


class MyNet(paddle.nn.Layer):
    def __init__(self):
        super(MyNet,self).__init__()
        self.resnet = paddle.vision.models.mobilenet_v2(pretrained=True, num_classes=0)
        self.flatten = paddle.nn.Flatten()
        self.linear = paddle.nn.Linear(1280, 4)

    def forward(self, img):
        y = self.resnet(img)
        y = self.flatten(y)
        out = self.linear(y)

        return out


model = MyNet()

weights = paddle.load('result.pdparams')
model.load_dict(weights)
model.eval()

def process(src_image_dir, output_filename):
    current_path = os.path.dirname(__file__)
    image_paths = glob.glob(os.path.join(src_image_dir, "*.jpg"))

    with open(os.path.join(current_path, output_filename), 'w') as f:
        for image_path in image_paths:
            image_name = image_path.split('/')[-1]
            img = cv2.imread(image_path)
            img = paddle.vision.transforms.resize(img, (512,512), interpolation='bilinear').transpose((2,0,1))/255
            img = paddle.to_tensor(img).unsqueeze(0).astype('float32')
            pre = model(img)
            pred_label = pre.argmax(-1).cpu().numpy()[0]
            f.write(f'{image_name} {pred_label}\n')
        f.close()


if __name__ == "__main__":
    assert len(sys.argv) == 3

    src_image_dir = sys.argv[1]
    output_filename = sys.argv[2]
    process(src_image_dir, output_filename)