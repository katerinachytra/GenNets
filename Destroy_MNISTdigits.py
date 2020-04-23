# Simple code for destroying MNIST digits, 
# Useful for image restoration tasks when something is missing in the image

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def destroy(train_x, label_x):
    # null random rows or columns
    i, j = np.min(train_x.shape),0
    while i+j >= np.min(train_x.shape):
        i, j = [item for item in np.random.randint([8,2], [27, 7], size=2)]
    mask = tf.constant(28*[i*[1.]+j*[0.]+(28-i-j)*[1.]])
    
    # null horizontally or or vertically, one always horizontally
    if np.random.normal(0, 1) >=0 or label_x == 1:
        mask = tf.transpose(mask)

    destroyed = mask*train_x   # element-wise multiplication
    return destroyed


def save_images(orig, destroyed):
    fig = plt.figure(figsize=(4, 4))
    for i in range(0,16,2):
        
        plt.subplot(4, 4, i+1)
        plt.imshow(orig[i, :, :], cmap='gray')
        plt.axis('off')
        plt.title('orig')
        
        plt.subplot(4, 4, i + 2)
        plt.imshow(destroyed[i, :, :], cmap='gray')
        plt.axis('off')   
        plt.title('destroyed')
        
    plt.tight_layout()
    plt.savefig('orig-destroyed.png')

# --DATASET----
digits_mnist = tf.keras.datasets.mnist
(train_images, train_label), (_,_) = digits_mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28).astype('float32')
# Normalizing the images to the range of [0., 1.]
train_images /= 255.

destroys = []

for train_x_orig, label in zip(train_images, train_label):
    train_x = destroy(train_x_orig, label)
    destroys.append(train_x)

destroys = np.asarray(destroys)
# prepare dataset for training
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, destroys))
save_images(train_images, destroys)
