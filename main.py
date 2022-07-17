import paddle

import numpy as np
import pandas as pd
import glob
import cv2


test_path = ''
train_path = ''
class MyDateset(paddle.io.Dataset):
    def __init__(self, pathTxt, mode = 'train'):
        super(MyDateset, self).__init__()
    
        with open(pathTxt,'r') as f:
            self.train_paths = f.readlines()
        self.mode = mode

    def __getitem__(self, index):
        item = self.train_paths[index].replace('\n','')
        # print(item)
        path, label = item.split(' ')
        img = cv2.imread('text_image_orientation/'+path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = int(label)
        
        img = paddle.vision.transforms.resize(img, (512,512), interpolation='bilinear')
        if self.mode == 'train':
            # 对图片进行resize，调整明暗对比度等参数
            
            if np.random.rand()<1/3:
                img = paddle.vision.transforms.adjust_brightness(img, np.random.rand()*2)
            else:
                if np.random.rand()<1/2:
                    img = paddle.vision.transforms.adjust_contrast(img, np.random.rand()*2)
                else:
                    img = paddle.vision.transforms.adjust_hue(img, np.random.rand()-0.5)

            img = img.transpose((2,0,1))
            img = img/255
            img = np.array(img).astype(np.float32)
            labels = [0,0,0,0]
            labels[label] = 1
            labels = np.array(labels).astype(np.float32)

            return img, labels
        else:
            img = img.transpose((2,0,1))
            img = img/255
            img = np.array(img).astype(np.float32)
            # labels = np.array(label).astype(np.float32)
            label = np.array([label])
            return img, label



    def __len__(self):
        return len(self.train_paths)



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
model.train()


params_info = paddle.summary(model,(1,3,512,512))
print(params_info)
# 需要接续之前的模型重复训练可以取消注释
# param_dict = paddle.load('./model.pdparams')
# model.load_dict(param_dict)




test_dataset=MyDateset(test_path,mode = 'test')
test_dataloader = paddle.io.DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=True,
    drop_last=False)


train_dataset=MyDateset(train_path,mode = 'train')
train_dataloader = paddle.io.DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    drop_last=False)




max_epoch=100
scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.0001, T_max=max_epoch)
opt = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())

now_step=0
for epoch in range(max_epoch):
    for step, data in enumerate(train_dataloader):
        now_step+=1

        img, label = data
        pre = model(img)
        # print(pre.shape,label.shape,'label')
        loss = paddle.nn.functional.square_error_cost(pre,label).mean()
        loss.backward()
        opt.step()
        opt.clear_gradients()
        if now_step%1000 == 0:
            model.eval()
            acc = 0
            for i, (img, label) in enumerate(test_dataloader):
                pre = model(img)
                # print(pre.shape,label.shape)
                acc += paddle.metric.accuracy(input=pre, label=label, k=1)
            print("epoch: {}, batch: {}, loss : {:.3f}, Acc: {:.3f}".format(epoch, step, loss.mean().numpy()[0],acc.cpu().numpy()[0]/(i+1)))
            model.train()
       
            paddle.save(model.state_dict(), 'result.pdparams')