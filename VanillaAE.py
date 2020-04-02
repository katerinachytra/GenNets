#import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Flatten, Dense, Reshape, UpSampling2D
import matplotlib.pyplot as plt
import PIL
import imageio
from IPython import display
import glob

lat_dim = 20

fashion_mnist = keras.datasets.fashion_mnist
(train_images, _), (test_images, _) = fashion_mnist.load_data()
input_shape = train_images.shape # (60000,28,28)
#train_images=train_images.reshape(train_images.shape[0], train_images.shape[1]*train_images.shape[2])
#input_shape = train_images.shape #
input_shape = (train_images.shape[0], input_shape[1]*input_shape[2],)
#input_shape = (input_shape[1]*input_shape[2], )
train_images = train_images.astype('float32')
# (training_features, _), (test_features, _) = tf.keras.datasets.mnist.load_data()
# train_images=train_images.reshape(train_images.shape[0], train_images.shape[1]*train_images.shape[2])
# test_images=test_images.reshape(test_images.shape[0], test_images.shape[1]*test_images.shape[2])
# training_dataset = tf.data.Dataset.from_tensor_slices(train_images)

# Encoder
inputs = Input(shape=(28, 28))
x = Flatten()(inputs)
#hidden = Dense(units=120,  activation=tf.nn.relu)(inps)
x = Dense(units=64,  activation=tf.nn.relu)(x)
latents = Dense(units=lat_dim,  activation=tf.nn.relu)(x)
encoder = tf.keras.Model(inputs=inputs, outputs=latents, name='encoder')

# Decoder
lats = Input(shape=(lat_dim,))
#y = Dense(units=49)(lats)
y=lats
y = Dense(units=28*28)(y)
y = Reshape((28, 28))(y)
#y = UpSampling2D((28*28/7, 28*28/7,))(y)
decoder = tf.keras.Model(lats, y, name='decoder')

outputs = decoder([encoder(inputs)])
VanillaAE = tf.keras.Model(inputs, outputs)

print(inputs.shape)
print(outputs.shape)
loss='mse'  #{'mse',mae,'binary_crossentropy','KLd',categorical_hinge}

VanillaAE.compile(optimizer='mse', loss=loss, metrics=['accuracy'])
epoch_no, batch_size = 5, 60
train_model = VanillaAE.fit(train_images, train_images, epochs=epoch_no, batch_size=batch_size)

test_loss, test_acc, = VanillaAE.evaluate(test_images,  test_images, verbose=2)
print('Test accuracy:', test_acc)
#print('Test prec:', test_prec)
file_weights = 'VanillaAE/weights/latdim=50/{loss:s}_batches{b:d}epochs{n:d}.h1'.format(loss=loss,n=epoch_no,b=batch_size)
VanillaAE.save_weights(file_weights)

#VanillaAE.load_weights(file_weights)
predictions = VanillaAE.predict(test_images)
#precision = tf.compat.v1.metrics.precision(test_images, predictions)
#print('Test precision:', precision)

fig, axes=plt.subplots(4, 4)
ax=axes.ravel()
i=0
while i<= (len(ax)-1):
  ax[i].axis('off')
  ax[i].set_title('test')
  ax[i+1].axis('off')
  ax[i+1].set_title('pred')
  ax[i].imshow(test_images[i])
  ax[i+1].imshow(predictions[i])
  i=i+2


plt.savefig('VanillaAE/images/{loss:s}_batches{b:d}epochs{n:d}ld_{ld:d}.png'.format(loss=loss,n=epoch_no,b=batch_size,ld=lat_dim))
plt.show()





