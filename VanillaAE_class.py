import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Flatten, Dense, UpSampling2D, Reshape
import matplotlib.pyplot as plt
import numpy as np

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, _), (test_images, _) = fashion_mnist.load_data()
input_shape = train_images.shape  # (60000,28,28)
input_shape = (train_images.shape[0], input_shape[1], input_shape[2],)
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
# Encoder
class autoencoder(tf.keras.Model):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                InputLayer(input_shape=(28, 28,)),
                Flatten(),
                Dense(units=64, activation=tf.nn.relu),
                Dense(units=24, activation=tf.nn.relu),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                InputLayer(input_shape=(24, )),
                Dense(units=49, activation=tf.nn.relu),
                Reshape((7, 7, 1)),
                UpSampling2D(size=(4, 4)),
            ]
        )

    def call(self, x):
        latents = self.encoder(x)
        decoded = self.decoder(latents)
        return decoded

VanillaAE = autoencoder()

VanillaAE.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

#------TRAIN ----
epoch_no, batch_size = 1, 120
# train_model = VanillaAE.fit(train_images, train_images, epochs=epoch_no, batch_size=batch_size)

# with open('VanillaAE_v{i:d}summary.txt'.format(i=1),'w') as fh:
#     VanillaAE.summary(print_fn=lambda x: fh.write(x + '\n'))

#tf.keras.utils.plot_model(VanillaAE, to_file='VanillaAEv{i:d}_2.png'.format(i=1),show_shapes=True)

file_weights = 'weights.h1'
# VanillaAE.save_weights(file_weights)
VanillaAE.load_weights(file_weights)
test_loss, test_acc, = VanillaAE.evaluate(test_images, test_images, verbose=2)
print('Test accuracy:', test_acc)

predictions = VanillaAE.predict(test_images)

fig, axes = plt.subplots(4, 4)
ax = axes.ravel()
i = 0
predictions = np.squeeze(predictions, axis=3)
print(predictions.shape)
while i <= (len(ax) - 1):
    ax[i].axis('off')
    ax[i].set_title('test')
    ax[i + 1].axis('off')
    ax[i + 1].set_title('pred')
    ax[i].imshow(test_images[i])
    ax[i + 1].imshow(predictions[i])
    i = i + 2

plt.savefig('VanillaAE/images/img{i:d}v3.png'.format(i=1))
plt.show()
