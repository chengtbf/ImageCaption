
import h5py
import numpy as np


# structure of h5 file ['test_set', 'train_set', 'validation_set']
file = h5py.File('/home/chengcheng/dataset/image_caption/feat.hdf5', 'r')
encoded_images = file['train_set']

print(file.keys())
print(encoded_images[0])
print(encoded_images[1])
print(encoded_images[2])
print(encoded_images[3])
print(encoded_images[4])
print(encoded_images[5])
print(encoded_images[6])
print(encoded_images[7])
print(np.array(encoded_images).shape)

encoded_images = file['valid_set']

print(encoded_images[0])
print(encoded_images[1])
print(np.array(encoded_images).shape)

encoded_images = file['test_set']

print(np.array(encoded_images).shape)


