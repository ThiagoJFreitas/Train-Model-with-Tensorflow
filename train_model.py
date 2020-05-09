import os
import os.path
from PIL import Image
import numpy as np
from numpy import asarray
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as kl
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

#%%
path_ = 'I:\\Empresas\\2stars\\IA\\Diagnostico_de_Radiografias\\Datasets\\01\\images_resized\\'
#%%
train_labels = np.load(path_ + 'numpy_files/train_labels.npy')
print(train_labels.shape)
train_images = np.load(path_ + 'numpy_files/train_images.npy')
print(train_images.shape)

#%%
from tensorflow.keras.applications.densenet import DenseNet121
img_in = kl.Input((224, 224, 3))
model = DenseNet121(include_top= False , # remove  the 3 fully-connected layers at the top of the network
                weights='imagenet',      # pre train weight
                input_tensor=img_in,
                input_shape=(224, 224, 3),
                pooling ='avg')
x = model.output
predictions = kl.Dense(2, activation="sigmoid", name="predictions")(x)    # fuly connected layer for predict class
model = Model(inputs=img_in, outputs=predictions)
callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5), keras.callbacks.ModelCheckpoint(filepath=path_ + 'model/best_model.h5', monitor='val_loss', save_best_only=True)]
opt = keras.optimizers.Adam(lr=0.001, epsilon = 1e-8, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False)
print(model.summary())

#%%
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, batch_size=32, validation_split=0.2, epochs=1, callbacks=callbacks)

#%%
model.save(path_ + 'model/model.h5')

#%%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training accuracy')
plt.legend()
plt.savefig(path_ + 'plots/accuracy.png')
plt.show()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training loss')
plt.legend()
plt.savefig(path_ + 'plots/loss.png')
plt.show()

#%%
train_images = []
train_labels = []
test_labels = np.load(path_ + 'numpy_files/test_labels.npy')
print(test_labels.shape)
test_images = np.load(path_ + 'numpy_files/test_images.npy')
print(test_images.shape)

#%%
predictions = model.predict(test_images)
predictions = abs(np.rint(predictions))
hits = 0
for i in range(len(predictions)):
  print(test_labels[i], "       ", predictions[i])
for i in range(len(predictions)):
  if (test_labels[i] == predictions[i]).all():
    hits += 1
accuracy = 100 * hits / 624
print('accuracy', accuracy)
model.evaluate(test_images, test_labels)

#%%
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(path_ + "modelTFlite/moses1.tflite", "wb").write(tflite_model)
