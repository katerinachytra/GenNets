import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import InputLayer, Flatten, Dense, Reshape, Conv2D, Conv2DTranspose
import matplotlib.pyplot as plt
import os

# AutoEncoder
class autoencoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                InputLayer(input_shape=(28, 28, 1)),
                Conv2D(filters=8,
                       kernel_size=(3, 3),
                       strides=(2, 2),
                       padding="same",
                       activation="relu",
                       use_bias=True,
                       kernel_initializer='glorot_uniform',
                       bias_initializer='zeros', ),
                Conv2D(filters=2,
                       kernel_size=(3, 3),
                       strides=(2, 2),
                       padding="same",
                       activation="relu",
                       use_bias=True,
                       kernel_initializer='glorot_uniform',
                       bias_initializer='zeros', ),
                Flatten(),
                Dense(units=latent_dim, activation=tf.nn.relu),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                InputLayer(input_shape=(latent_dim,)),
                Dense(units=49, activation=tf.nn.relu),
                Reshape((7, 7, 1)),
                Conv2DTranspose(1, (3, 3), padding='same', strides=2, activation='relu', use_bias=True,
                                kernel_initializer='glorot_uniform',
                                bias_initializer='zeros'),
                Conv2DTranspose(2, (3, 3), padding='same', strides=2, activation='relu', use_bias=True,
                                kernel_initializer='glorot_uniform',
                                bias_initializer='zeros'),
                Conv2DTranspose(8, (3, 3), padding='same', strides=1, activation='relu', use_bias=True,
                                kernel_initializer='glorot_uniform',
                                bias_initializer='zeros'),
                Conv2DTranspose(1, (3, 3), padding='same', strides=1, activation='relu', use_bias=True,
                                kernel_initializer='glorot_uniform',
                                bias_initializer='zeros'),
            ]
        )

    def call(self, x):
        latents = self.encoder(x)
        decoded = self.decoder(latents)
        return decoded
# ---------------------------------------------------------------------------------------------
ConvAE = autoencoder(latent_dim=24)
ConvAE.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

keras.utils.plot_model(ConvAE, to_file='ConvAE_v{i:d}.png'.format(i=ver), show_shapes=True)

# --DATASET---
digits_mnist = keras.datasets.mnist
(train_images, _), (test_images, _) = digits_mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
# Normalizing the images to the range of [0., 1.]
train_images /= 255.
test_images /= 255.
# --------------------------------------------------------------------
# ----VISUALIZATION--TENSORBOARD---------------
import datetime
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
from tensorboard import program

tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', log_dir])
url = tb.launch()
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# -----------------
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# Callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
# --TRAINING----
epoch_no, batch_size = 5, 100
train_model = ConvAE.fit(train_images, train_images, epochs=epoch_no, batch_size=batch_size,
                         callbacks=[tb_callback, cp_callback])

file_weights = 'weights_ConvAE.h1'
ConvAE.save_weights(file_weights)
# ConvAE.load_weights(file_weights)
# ---Save the entire model as a SavedModel---
ConvAE.save('saved_model/ConvAE')
# ---Save Summary to the file------------------------------------
with open('summary_ConvAE.txt', 'w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    ConvAE.summary(print_fn=lambda x: fh.write(x + '\n'))
# ----EVALUATE----------
test_loss, test_acc, = ConvAE.evaluate(test_images, test_images, verbose=2)
print('Test accuracy:', test_acc)
# ---PREDICTION---
predictions = ConvAE.predict(test_images)

predictions = np.squeeze(predictions, axis=3)
print(predictions.shape)
test_images = np.squeeze(test_images, axis=3)

fig, axes = plt.subplots(4, 4)
ax = axes.ravel()
i = 0
while i <= (len(ax) - 1):
    ax[i].axis('off')
    ax[i].set_title('orig')
    ax[i + 1].axis('off')
    ax[i + 1].set_title('pred')
    ax[i].imshow(test_images[i])
    ax[i + 1].imshow(predictions[i])
    i = i + 2

plt.savefig('digits.png')
plt.show()
