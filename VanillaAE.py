import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Flatten, Dense, Reshape
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, _), (test_images, _) = fashion_mnist.load_data()
input_shape = train_images.shape # (60000,28,28)
input_shape = (train_images.shape[0], input_shape[1]*input_shape[2],)

test_images = train_images.astype('float32')
test_images = test_images.astype('float32')

#Vanilla AUTOENCODER
lat_dim = 20
# Encoder
inputs = Input(shape=(28, 28))
x = Flatten()(inputs)
x = Dense(units=64,  activation=tf.nn.relu)(x)
latents = Dense(units=lat_dim,  activation=tf.nn.relu)(x)
encoder = tf.keras.Model(inputs=inputs, outputs=latents, name='encoder')

# Decoder
lats = Input(shape=(lat_dim,))
y = Dense(units=28*28)(lats)
y = Reshape((28, 28))(y)
decoder = tf.keras.Model(lats, y, name='decoder')

outputs = decoder([encoder(inputs)])
VanillaAE = tf.keras.Model(inputs, outputs)

#COMPILE
VanillaAE.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
#TRAIN
epoch_no, batch_size = 5, 60
train_model = VanillaAE.fit(train_images, train_images, epochs=epoch_no, batch_size=batch_size)

test_loss, test_acc, = VanillaAE.evaluate(test_images,  test_images, verbose=2)
print('Test accuracy:', test_acc)

file_weights = 'weights_batches{b:d}epochs{n:d}.h1'.format(n=epoch_no,b=batch_size)
VanillaAE.save_weights(file_weights)

#VanillaAE.load_weights(file_weights)
predictions = VanillaAE.predict(test_images)

#PLOT PREDICTIONS AND ORIGINAL IMAGES
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
plt.savefig('VanillaAE{b:d}epochs{n:d}ld_{ld:d}.png'.format(n=epoch_no,b=batch_size,ld=lat_dim))
plt.show()
