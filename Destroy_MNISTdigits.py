import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import h5py


def destroy(train_x, label_x):
    # null random rows or columns
    i, j = np.min(train_x.shape), 0
    while i + j >= np.min(train_x.shape):
        i = np.random.randint(8, 27, size=1)[0] # radek
        j = np.random.randint(2, 7, size=1)[0] # pocet radku
    mask = np.ones(train_x.shape)
    mask[i:i+j, :] = 0
    # null horizontally or vertically, one always horizontally
    if np.random.normal(0, 1) >= 0 or label_x == 1:
        mask = np.transpose(mask)
    destroyed = mask * train_x  # element-wise multiplication
    return destroyed


def save_images(orig, destroyed):
    fig = plt.figure(figsize=(4, 4))

    for i in range(0, 16, 2):
        plt.subplot(4, 4, i + 1)
        plt.imshow(orig[i, :, :], cmap='gray')
        plt.axis('off')
        plt.title('orig')

        plt.subplot(4, 4, i + 2)
        destroyed = np.asarray(destroyed)
        plt.imshow(destroyed[i, :, :], cmap='gray')
        plt.axis('off')
        plt.title('destroyed')
    plt.tight_layout()
    plt.savefig('orig-destroyed.png')


start_time = time.time()
# --DATASET----
digits_mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = digits_mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28).astype('float32')
# Normalizing the images to the range of [0., 1.]
train_images /= 255.

TRAIN_BUF = 10000 # 10 000
train_images = train_images[:TRAIN_BUF]
train_labels = train_labels[:TRAIN_BUF]

# create hdf5 file, which will contain all dataset
f = h5py.File('destroy_dset.hdf5', 'a') # nelze 'a' - musela bych oštřit, aby groups nemely stejna jmena

grp_names = sorted(int(item) for item in list(f.keys()))
print(grp_names)
grp_name = int(grp_names[-1])+2
print(grp_name)
i = 0
for train_x_orig, label in zip(train_images, train_labels):
    start_loop = time.time()
    train_x = destroy(train_x_orig, label)
    print(grp_name)
    grp = f.create_group(str(grp_name))
    grp.attrs['label'] = label
    dset1 = grp.create_dataset('destr', data=train_x)
    dset2 = grp.create_dataset('orig', data=train_x_orig)
    print('step: ', i)
    i = i + 1
    grp_name = grp_name+1

def printname(name):
    print(name)

# f.visit(printname) # vypise cely dataset - zrejma struktura


end_time = time.time()
print('Time to elapse the destruction: ', round(end_time - start_time, 3), 's')
exit()
print(f['0'])
dgrp = f['0']
print(dgrp.attrs['label'])
dset = dgrp['destr']
plt.imshow(dset)
plt.show()


# save_images(train_images,destroys)

