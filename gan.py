from math import sqrt
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.losses import binary_crossentropy
from networks import make_generator, make_discriminator

# try 
# - adjusting learning rates
# - adjusting neurons per layer
# - adjusting number of layers
# - add dropout

BATCH_SIZE = 64
NUM_EPOCHS = 60

GENERATOR_LEARN_RATE = 2e-5
DISCRIMINATOR_LEARN_RATE = 1e-5

# for displaying progress
NUM_SAMPLES = 9
assert int(sqrt(NUM_SAMPLES)) == sqrt(NUM_SAMPLES), 'NUM_SAMPLES must be a perfect square'

# generates and displays samples from the generator
def show_progress(generator, subplots, epoch=0, batch=0):
  fig, axes = subplots
  test_imgs = generator(tf.random.normal([9, 100]), training=False)
  test_imgs_2D = np.reshape(test_imgs, (9, 28, 28))

  if epoch > 0:
    fig.suptitle('Epoch {}, Batch {} samples'.format(epoch, batch))

  for img, axis in zip(test_imgs_2D, np.ndarray.flatten(axes)):
    axis.imshow(img, cmap='gray')

  plt.pause(0.05)
  plt.show(block=False)

# load mnist data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# pick only images depicting an 8 and normalize [0, 1]
x_only_eights = np.array([x for x, y in zip(x_train, y_train) if y == 8]) / 255

# flatten each image to vector
# x_only_eights = x_only_eights.reshape(x_only_eights.shape[0], x_only_eights.shape[1] * x_only_eights.shape[2])

# set up generator
generator = make_generator()
generator_optimizer = tf.keras.optimizers.Adam(GENERATOR_LEARN_RATE)

# set up discriminator
discriminator = make_discriminator()
discriminator_optimizer = tf.keras.optimizers.Adam(DISCRIMINATOR_LEARN_RATE)

# progress check setup
plt.ion()
subplots = plt.subplots(int(sqrt(NUM_SAMPLES)), int(sqrt(NUM_SAMPLES)))

# train
num_batches = np.shape(x_only_eights)[0] // BATCH_SIZE
for epoch in range(NUM_EPOCHS):
  print('Training epoch', epoch + 1, 'of', NUM_EPOCHS, '...')
  for batch in range(num_batches):
    x_slice = x_only_eights[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE]
    noise = tf.random.normal([len(x_slice), 100], 0, 1)
    with tf.GradientTape() as discriminator_tape:
      # generate fake image from noise
      generator_output = generator(noise, training=False)

      # feed real image to discriminator
      real_guess = discriminator(x_slice, training=True)[0]
      # feed fake image to discriminator
      fake_guess = discriminator(generator_output, training=True)[0]

      # calculate loss
      discriminator_loss = (binary_crossentropy(real_guess, np.ones_like(real_guess)) + binary_crossentropy(fake_guess, np.zeros_like(fake_guess))) / 2

    discriminator_gradient = discriminator_tape.gradient(discriminator_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(discriminator_gradient, discriminator.trainable_variables))

    with tf.GradientTape() as generator_tape:
      # generate fake image from noise
      generator_output = generator(noise, training=True)

      # feed fake image to discriminator
      fake_guess = discriminator(generator_output, training=False)[0]
      generator_loss = binary_crossentropy(fake_guess, np.ones_like(fake_guess))
      
    generator_gradient = generator_tape.gradient(generator_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradient, generator.trainable_variables))

    print('g-loss:', float(generator_loss), ' d-loss:', float(discriminator_loss))
  
  # show generator progress
  show_progress(generator, subplots, epoch=epoch + 1, batch=batch+1)

#test
subplots[0].suptitle('Final Results')
show_progress(generator, subplots)

plt.ioff()
plt.show()