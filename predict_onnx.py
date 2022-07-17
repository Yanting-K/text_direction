
import os
import sys
import glob
import cv2
import onnx 
import onnxruntime as ort
import numpy as np


def process(src_image_dir, output_filename):
    current_path = os.path.dirname(__file__)
    image_paths = glob.glob(os.path.join(src_image_dir, "*.jpg"))

    with open(os.path.join(current_path, output_filename), 'w') as f:
        for image_path in image_paths:
            image_name = image_path.split('/')[-1]
            img = cv2.imread(image_path)
            img = cv2.resize(img,(512,512), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_data = np.transpose(img,(2,0,1))
            input_data = input_data/255.
            input_data = input_data.reshape([1,3,512,512]).astype('float32')
            sess = ort.InferenceSession('result.onnx',providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])


            input_name = sess.get_inputs()[0].name
            result = sess.run(None,{input_name:input_data})
            result = np.reshape(result,[1,-1])
            index = np.argmax(result)
    
            print(index)
            f.write(f'{image_name} {index}\n')
        f.close()


if __name__ == "__main__":
    assert len(sys.argv) == 3

    src_image_dir = sys.argv[1]
    output_filename = sys.argv[2]
    process(src_image_dir, output_filename)






