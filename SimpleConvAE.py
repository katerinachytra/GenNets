import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Flatten, Dense, Reshape, Conv2D, Conv2DTranspose
import matplotlib.pyplot as plt

digits_mnist = keras.datasets.mnist
(train_images, _), (test_images, _) = digits_mnist.load_data()
input_shape = train_images.shape  # (60000,28,28)

input_shape = (input_shape[1], input_shape[2], 1,)
print('input shape:', input_shape)
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images = np.expand_dims(train_images, axis=3)  # without extension - model.fit error
test_images = np.expand_dims(test_images, axis=3)
# ---Architecture of the encoder --- 
# === ENCODER ====
inputs = Input(shape=input_shape)
# Conv1 (using stride = 2 instead of MaxPooling layer)
x = Conv2D(filters=8,
           kernel_size=(3, 3),
           strides=(2, 2),
           padding="same",
           activation="relu",
           use_bias=True,
           kernel_initializer='glorot_uniform',
           bias_initializer='zeros', )(inputs)
# Conv2
x = Conv2D(filters=2,
           kernel_size=(3, 3),
           strides=(2, 2),
           padding="same",
           activation="relu",
           use_bias=True,
           kernel_initializer='glorot_uniform',
           bias_initializer='zeros')(x)
x = Flatten()(x)
# Dense
x = Dense(units=24)(x)
# ==== DECODER  =====
# Dense
y = Dense(7*7)(x)
y = Reshape((7, 7, 1))(y)
y = Conv2DTranspose(1, (3, 3), padding='same', strides=2, activation='relu', 
                    use_bias=True, kernel_initializer='glorot_uniform',
           bias_initializer='zeros')(y)
# Conv1
y = Conv2DTranspose(2, (3, 3), padding='same', strides=2, activation='relu', 
                    use_bias=True, kernel_initializer='glorot_uniform',
           bias_initializer='zeros')(y)
# Conv2
y = Conv2DTranspose(8, (3, 3), padding='same', strides=1, activation='relu', 
                    use_bias=True, kernel_initializer='glorot_uniform',
           bias_initializer='zeros')(y)
# Conv3
y = Conv2DTranspose(1, (3, 3), padding='same', activation='relu', 
                    use_bias=True, kernel_initializer='glorot_uniform',
           bias_initializer='zeros')(y) #sigmoid - output pixels in range [0,255]
# ----------------------
autoencoder = tf.keras.Model(inputs, y)
keras.utils.plot_model(autoencoder, to_file='CNN_AEv{i:d}.png'.format(i=ver),show_shapes=True)
# ---COMPILE MODEL---
autoencoder.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])

# ----PRINT MODEL SUMMARY INTO A FILE -----
with open('CNN_AE_v{i:d}summary.txt'.format(i=ver),'w') as fh:
    autoencoder.summary(print_fn=lambda x: fh.write(x + '\n'))

# ------TRAIN THE MODEL
epoch_no, batch_size = 5, 120
train_model = autoencoder.fit(train_images, train_images, epochs=epoch_no, batch_size=batch_size)

file_weights = 'weights.h1'
autoencoder.save_weights(file_weights)
#autoencoder.load_weights(file_weights)

# ---EVALUATION ---
test_loss, test_acc, = autoencoder.evaluate(test_images, test_images, verbose=2)
print('Test accuracy:', test_acc)
predictions = autoencoder.predict(test_images)

test_images = np.squeeze(test_images, axis=3)
predictions = np.squeeze(predictions, axis=3)
print(predictions.shape)
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

plt.savefig('digits{i:d}.png'.format(i=ver))
plt.show()
