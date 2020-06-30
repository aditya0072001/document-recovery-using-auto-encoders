#importing required librabries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split


train_images_path = 'data/train'
train_cleaned_path = 'data/train_cleaned'
test_images_path  = './test'
train_images = sorted(os.listdir(train_images_path))
train_cleaned = sorted(os.listdir(train_cleaned_path))
test_images = sorted(os.listdir(test_images_path))


X = []
y = []

for img in train_images:
    img_path = os.path.join(train_images_path, img)
    im = load_img(img_path, color_mode = 'grayscale', target_size = (540, 260))
    im = img_to_array(im).astype('float32')/255
    X.append(im)
for img in train_cleaned:
    img_path = os.path.join(train_cleaned_path, img)
    im = load_img(img_path, color_mode = 'grayscale', target_size = (540, 260))
    im = img_to_array(im).astype('float32')/255
    y.append(im)
    
X = np.array(X)
y = np.array(y)

print('Shape of X: ', X.shape)
print('Shape of y: ', y.shape)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.1, random_state = 42)

input_img = Input(shape=(540,260,1)) 

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (7, 7, 32)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='min', restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.005)

autoencoder.compile(optimizer='Adam', loss='binary_crossentropy',metrics=['accuracy'])

history = autoencoder.fit(X_train, y_train, epochs = 200, batch_size = 8, validation_data = (X_valid, y_valid), verbose = True, callbacks = [es,rlr])

autoencoder.save('model.h5')