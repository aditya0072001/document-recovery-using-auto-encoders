from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
model=load_model('model.h5')

test_images_path  = './internet'
test_images = sorted(os.listdir(test_images_path))
X_test = []
for img in test_images:
    img_path = os.path.join(test_images_path, img)
    im = load_img(img_path, color_mode = 'grayscale', target_size = (540, 260))
    im = img_to_array(im).astype('float32')/255
    X_test.append(im)
    
X_test = np.array(X_test)

hehe=model.predict(X_test)


for i in range(8):
    plt.imshow(cv2.resize(X_test[i],(540,260)),cmap='gray')
    plt.show()
    plt.imshow(cv2.resize(hehe[i],(540,260)),cmap='gray')
    plt.show()
