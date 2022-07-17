# 百度网盘AI大赛——图像处理挑战赛：文档图像方向识别
> 此baseline提供paddle版本及如何转换为onnx的模型并提交，并不注重性能。
敬请知晓~

### 数据准备
下载paddle官方提供的数据
```
wget https://paddleclas.bj.bcebos.com/data/PULC/text_image_orientation.tar
tar -xf text_image_orientation.tar
```


### 训练模型
进入main.py后，修改**test_path**以及**train_path**的路径，即刚刚解压的text_image_orientation/test_list.txt，train_path为train_list.txt

然后运行
```
python train.py
```

### 测试模型
训练模型结束后，根目录下会有一个**result.pdparams**，如果修改了路径可以在**predict.py**的25行修改路径。
运行
```
python predict.py test_A/images predict.txt
```
运行结束后得到predict.txt文档， 内容如下：
```
4f7dc1142d1da042c5f8c1cd65f45a58.jpg 2
5172666a52c74cf8c07e74ca10905e71.jpg 2
4a3db2f73b578639470c775b5da369f9.jpg 2
ed6bb52a3f69f170a37b4f139f221e48.jpg 0
a011189135e2d81c5c9cadbd37162e79.jpg 1
2d9bf797cea59c014b87cad2dca15f42.jpg 0
04f5597fe9e63eb9d31b2ff5a0e518b3.jpg 1
···················
```
如果走到这一步的话就说明paddle版本已经跑通了，在命令行中输入
```
zip -r result.zip predict.py result.pdparams 
```
将这个zip打包上传即可

### 测试ONNX版本
ONNX需要使用静态图的模型转换，首先讲训练完的动态图转为静态图
```
python lite.py --model_path result.pdparams 
```
然后会看到在output文件夹里面有三个文件，这就是静态图的模型文件。

继续在命令行输入
```
paddle2onnx --model_dir output --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 11 --save_file result.onnx
```
运行结束后会看到在根目录有一个**result.onnx**文件。
运行
```
python predict_onnx.py  test_A/images predict.txt
```
运行完之后得到一个predict.txt的文件，这个文件内容应如上相同。

因为上传文件需要**predict.py**的预测脚本，所以如果上传onnx版本的话，需要将predict_onnx.py改为predict.py再上传！
```
mv predict_onnx.py predict.py # 这一步会把paddle版本的脚本覆盖掉，提前将paddle的预测脚本改个名~
zip -r result.zip predict.py result.pdparams 
```
