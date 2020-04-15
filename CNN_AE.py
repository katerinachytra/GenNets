import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Flatten, Dense, Reshape, Conv2D, UpSampling2D, MaxPooling2D, Conv2DTranspose
import matplotlib.pyplot as plt
import datetime

# ----Architecture version
ver = 13
# ---DATASET---------
# fashion_mnist = keras.datasets.fashion_mnist
digits_mnist = keras.datasets.mnist
# (train_images, _), (test_images, _) = fashion_mnist.load_data()
(train_images, _), (test_images, _) = digits_mnist.load_data()
input_shape = train_images.shape  # (60000,28,28)
image_shape = (input_shape[1], input_shape[2], 1,)

input_shape = image_shape
print('input shape:', input_shape)
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images = np.expand_dims(train_images, axis=3)  # without extension - model.fit error
test_images = np.expand_dims(test_images, axis=3)
# === ENCODER ====
inputs = Input(shape=input_shape)
# Conv1
x = Conv2D(filters=8,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding="same",
           activation="relu",
           use_bias=True,
           kernel_initializer='glorot_uniform',
           bias_initializer='zeros', )(inputs)
x = MaxPooling2D((2, 2))(x)
# Conv2
x = Conv2D(filters=2,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding="same",
           activation="relu",
           use_bias=True,
           kernel_initializer='glorot_uniform',
           bias_initializer='zeros')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
# Dense
x = Dense(units=49)(x)
# encoder = tf.keras.Model(inputs=inputs, outputs=x, name='encoder')

# ==== DECODER  =====
# Dense
y = Dense(7 * 7)(x)
y = Reshape((7, 7, 1))(y)
y = UpSampling2D((2, 2))(y)
# Conv1
y = Conv2DTranspose(2, (3, 3), padding='same', activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
                    bias_initializer='zeros')(y)
y = UpSampling2D((2, 2))(y)
# Conv2
y = Conv2DTranspose(8, (3, 3), padding='same', activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
                    bias_initializer='zeros')(y)
# Conv3
y = Conv2DTranspose(1, (3, 3), padding='same', activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
                    bias_initializer='zeros')(y)  # sigmoid - output pixels in range [0,255]
print('decoder output shape', y.shape)
# decoder = tf.keras.Model(lats, y, name='decoder')

autoencoder = tf.keras.Model(inputs, y)
autoencoder.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])

# keras.utils.plot_model(autoencoder, to_file='CNN_AEv{i:d}.png'.format(i=ver),show_shapes=True)
# autoencoder.save('CNN_AE_v{i:d}'.format(i=ver))
# =================================================================
# Open the file
# with open('CNN_AE_v{i:d}summary.txt'.format(i=ver),'w') as fh:
# Pass the file handle in as a lambda function to make it callable
# autoencoder.summary(print_fn=lambda x: fh.write(x + '\n'))

# ----VISUALIZATION-----------------
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
from tensorboard import program

tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', log_dir])
url = tb.launch()

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=.5)
# ----TRAIN-------------------------
epoch_no, batch_size = 5, 120
train_model = autoencoder.fit(train_images, train_images, epochs=epoch_no, batch_size=batch_size,
                              callbacks=[tensorboard_callback])

# ckpt = tf.train.Checkpoint(optimizer='Adam', model = autoencoder)
# manager = tf.train.CheckpointManager(ckpt,'./tf_ckpts', max_to_keep=3)
# save model
# tf.saved_model.save(autoencoder)
# save_path = manager.save()
# print('Saved checkpoint for step {int(step)}: {save_path}')

file_weights = 'weights.h1'
autoencoder.save_weights(file_weights)
# autoencoder.load_weights(file_weights)

writer = tf.summary.create_file_writer("mylogs")
with writer.as_default():
     # other model code would go here
     tf.summary.scalar("mse", 0.5, step=1)
     writer.flush()


# -----EVALUATION-----------------
test_loss, test_acc, = autoencoder.evaluate(test_images, test_images, verbose=2)
print('Test accuracy:', test_acc)
predictions = autoencoder.predict(test_images)

test_images = np.squeeze(test_images, axis=3)
predictions = np.squeeze(predictions, axis=3)
print('predictions shape', predictions.shape)
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

plt.savefig('CNN_AE/images/digits{i:d}v4.png'.format(i=ver))
plt.show()
