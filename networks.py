import keras
from keras import layers

def make_generator():
  model = keras.models.Sequential()
  model.add(layers.InputLayer(input_shape=(784,)))

  model.add(layers.Dropout(0.3))
  model.add(layers.Dense(275, activation=layers.LeakyReLU(alpha=0.2)))
  model.add(layers.Dense(512, activation=layers.LeakyReLU(alpha=0.2)))
  model.add(layers.Dense(784, activation='sigmoid'))

  return model

def make_discriminator():
  model = keras.models.Sequential()
  model.add(layers.InputLayer(input_shape=(784,)))

  model.add(layers.Dropout(0.3))
  model.add(layers.Dense(400, activation=layers.LeakyReLU(alpha=0.2)))
  model.add(layers.Dropout(0.3))
  model.add(layers.Dense(300, activation=layers.LeakyReLU(alpha=0.2)))
  model.add(layers.Dropout(0.3))
  model.add(layers.Dense(1, activation='sigmoid'))

  return model