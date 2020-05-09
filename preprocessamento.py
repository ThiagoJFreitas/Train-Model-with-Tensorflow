import os
import os.path
from PIL import Image
import numpy as np
from numpy import asarray

# %% Resize images
origin = 'I:\\Empresas\\2stars\\IA\\Diagnostico_de_Radiografias\\Datasets\\01\\chest_xray\\chest_xray\\train\\NORMAL'
destiny = 'I:\\Empresas\\2stars\\IA\\Diagnostico_de_Radiografias\\Datasets\\01\\images_resized\\train\\NORMAL\\'
# n = 9000
# count = 0
valid_images = [".jpeg"]
for f in sorted(os.listdir(origin)):
  if '._' in f:
    continue
  ext = os.path.splitext(f)[1]
  if ext.lower() not in valid_images:
    continue
  img = Image.open(os.path.join(origin, f))
  fname = img.filename
  img = img.convert("RGB")
  res_img = img.resize((224, 224))
  head, tail = os.path.split(fname)
  pth = destiny + tail
  res_img.save((pth.strip()))
  # count += 1
  # if count == n:
  #     break

  # %% Rename images
  origin = 'I:\\Empresas\\2stars\\IA\\Diagnostico_de_Radiografias\\Datasets\\01\\images_resized\\test\\NORMAL'
  destiny = 'I:\\Empresas\\2stars\\IA\\Diagnostico_de_Radiografias\\Datasets\\01\\images_resized\\test\\'
  # n = 9000
  count = 0
  valid_images = [".jpeg"]
  for f in sorted(os.listdir(origin)):
    if '._' in f:
      continue
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
      continue
    img = Image.open(os.path.join(origin, f))
    img = img.convert("RGB")
    fname = 'NORMAL' + str(count) + '.jpeg'
    pth = destiny + fname
    img.save((pth.strip()))
    count += 1
    # if count == n:
    #     break

    #%%
path_ = 'I:\\Empresas\\2stars\\IA\\Diagnostico_de_Radiografias\\Datasets\\01\\images_resized\\'
dir_train = path_ + 'test'
train_images = []
valid_images = [".jpeg"]
for f in sorted(os.listdir(dir_train)):
  ext = os.path.splitext(f)[1]
  if ext.lower() not in valid_images:
    continue
  img = Image.open(os.path.join(dir_train, f))
  img = img.convert("RGB")
  img = np.array(img)
  img = img[..., :3]
  train_images.append(img)

train_images = np.array(train_images).astype("float32") / 255
np.save(path_ + 'numpy_files\\test_images', train_images)  # change here

#%%
path_ = 'I:\\Empresas\\2stars\\IA\\Diagnostico_de_Radiografias\\Datasets\\01\\images_resized\\'
dir_ = path_ + 'train'
labels = []
valid_images = [".jpeg"]
for f in sorted(os.listdir(dir_)):
  ext = os.path.splitext(f)[1]
  if ext.lower() not in valid_images:
    continue
  label = [0, 0]
  if 'NORMAL' in f:
    label[0] = 1
  else:
    label[1] = 1
  labels.append(label)

np.save(path_ + 'numpy_files\\train_labels', labels)  # change here

#%%

