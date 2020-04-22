import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Flatten, Dense, Reshape, Conv2D, Conv2DTranspose
import matplotlib.pyplot as plt
import time
from IPython import display
import datetime
import os
from tensorboard import program


# Convolutional Variational AutoEncoder
class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                InputLayer(input_shape=(28, 28, 1)),
                Conv2D(16, (3, 3), strides=2, padding="same", activation='relu', use_bias=True,
                       kernel_initializer='glorot_uniform', bias_initializer='zeros', ),
                Conv2D(8, (3, 3), strides=2, padding="same", activation="relu", use_bias=True,
                       kernel_initializer='glorot_uniform', bias_initializer='zeros', ),
                Conv2D(4, (3, 3), strides=1, padding="same", activation="relu", use_bias=True,
                       kernel_initializer='glorot_uniform', bias_initializer='zeros', ),
                Flatten(),
                Dense(latent_dim + latent_dim),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                InputLayer(input_shape=(latent_dim,)),
                Dense(units=49, activation=tf.nn.relu),
                Reshape((7, 7, 1)),
                Conv2DTranspose(4, (3, 3), padding='same', strides=1, activation='relu', use_bias=True,
                                kernel_initializer='glorot_uniform', bias_initializer='zeros'),
                Conv2DTranspose(8, (3, 3), padding='same', strides=2, activation='relu', use_bias=True,
                                kernel_initializer='glorot_uniform', bias_initializer='zeros'),
                Conv2DTranspose(16, (3, 3), padding='same', strides=2, activation='relu', use_bias=True,
                                kernel_initializer='glorot_uniform', bias_initializer='zeros'),
                Conv2DTranspose(1, (3, 3), padding='same', strides=1, activation='relu', use_bias=True,
                                kernel_initializer='glorot_uniform', bias_initializer='zeros'),
            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))  # N(0,I), shape=(batch, latent_dim)
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        z_mean, z_logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return z_mean, z_logvar

    def reparametrize(self, z_mean, z_logvar):
        eps = tf.random.normal(shape=z_mean.shape)
        return z_mean + eps * tf.exp(z_logvar * .5)

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    # PLOT MODEL and PRINT SUMMARY
    def plotModel(self):
        tf.keras.utils.plot_model(self.encoder, to_file='encoder.png')
        tf.keras.utils.plot_model(self.decoder, to_file='decoder.png')
        with open('encoder_summary.txt', 'w') as fh:
            self.encoder.summary(print_fn=lambda x: fh.write(x + '\n'))
        with open('decoder_summary.txt', 'w') as fh:
            self.decoder.summary(print_fn=lambda x: fh.write(x + '\n'))
# ---------------------------------------------------------------------------------------------
optimizer = tf.keras.optimizers.Adam(1e-4)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


@tf.function
def compute_loss(model, x):
    z_mean, z_logvar = model.encode(x)
    z = model.reparametrize(z_mean, z_logvar)
    y = model.decode(z)

    # cross_ent = tf.nn.cross_entropy_with_logits(logits=x_logit, labels=x)
    mse = tf.keras.losses.mse(x, y)
    # logpx_z = tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpx_z = -tf.reduce_sum(mse) # minus because minimizing inherently, but in fact I need to maximize in the expression below
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, z_mean, z_logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)  # logqz_x - logpz = kl divergence, expression inside () we want to maximize thus minus


@tf.function
def compute_apply_gradients(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

epochs = 5
latent_dim = 24
num_examples_to_generate = 16

random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])
model = VAE(latent_dim)

tf.saved_model.save(model, 'vae_save/')
# model = tf.saved_model.load('logs/models/')

# model.plotModel()

def generate_and_save_images(model, epoch, test_input):
    predictions = model.sample(test_input)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')
        plt.tight_layout()
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()


generate_and_save_images(model, 0, random_vector_for_generation)

# TENSORBOARD
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "logs/gradient_tape/"
train_log_dir = log_dir + current_time + '/train'
test_log_dir = log_dir + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)
tf.summary.trace_on(graph=True, profiler=True)
# tb = program.TensorBoard()
# tb.configure(argv=[None, '--logdir', 'logs/gradient_tape/'])
# url = tb.launch() # does not work, launch it from terminal typing tensorboard --logdir 'logs/gradient_tape/'

# --DATASET----
digits_mnist = tf.keras.datasets.mnist
(train_images, _), (test_images, _) = digits_mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
# Normalizing the images to the range of [0., 1.]
train_images /= 255.
test_images /= 255.

TRAIN_BUF = 60000
BATCH_SIZE = 100

TEST_BUF = 10000
# create batches and shuffle the dataset
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
# ---
# CHECKPOINTS
ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(ckpt,'./tf_ckpts',max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print('Restored from {}', format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

 # ---
# WEIGHTS
if os.path.exists('vae_weights.h5'):
    model.load_weights('vae_weights.h5')
    print('Previous weights loaded.')
else:
    print('No previous weights.')

# --TRAINING-----
for epoch in range(1, epochs + 1):
    start_time = time.time()
    loss = tf.keras.metrics.Mean()
    for train_x in train_dataset:
        compute_apply_gradients(model, train_x, optimizer)
        loss(compute_loss(model, train_x))
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', -loss.result(), step=epoch)
    end_time = time.time()

    if epoch % 1 == 0:  # that means test after every epoch
        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            loss(compute_loss(model, test_x))
        elbo = -loss.result()  # evidence lower bound (maximized during training)
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', elbo, step=epoch)
        display.clear_output(wait=False)
        print('Epoch: {}, Test set ELBO: {}, '
              'time elapse for current epoch {}'.format(epoch, elbo, end_time - start_time))
        generate_and_save_images(model, epoch, random_vector_for_generation)

model.save_weights('vae_weights.h5', overwrite=True)
