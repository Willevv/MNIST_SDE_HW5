import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)

(training_images, training_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
depth = 10
training_labels = tf.one_hot(training_labels, depth)    # transforming the label data into one-hot vector form
test_labels = tf.one_hot(test_labels, depth)

print('Training Images Dataset Shape: {}'.format(training_images.shape))
print('No. of Training Images Dataset Labels: {}'.format(len(training_labels)))
print('Test Images Dataset Shape: {}'.format(test_images.shape))
print('No. of Test Images Dataset Labels: {}'.format(len(test_labels)))

K = 32    # K is the number of nodes in the first hidden layer
dt = 0.002    # dt is the learning rate for SGD method
N_training_set = 60000
Nbatch = 10
EPOCHS = 10
validation_split_ratio = 0.1
M_step = EPOCHS * N_training_set * (1 - validation_split_ratio) / Nbatch
T = M_step * dt
#print('Number of SGD steps M=', M_step)

xmean = np.mean(training_images)
xstd = np.sqrt(np.var(training_images))
training_images = (training_images - xmean) / xstd    # Normalization of training_data
test_images = (test_images - xmean) / xstd    # Normalization of test_data

#tf.random.set_seed(123)

input_data_shape = (28, 28)
hidden_activation_function = tf.keras.activations.relu
output_activation_function = tf.keras.activations.softmax
nn_model = keras.Sequential()
nn_model.add(layers.Flatten(input_shape=input_data_shape, name='Input_layer'))    # layers.Flatten maps the input tensor onto a vector

nn_model.add(layers.Dense(K,
                          activation=hidden_activation_function,
                          use_bias=True, 
                          kernel_initializer='random_normal',
                          bias_initializer='zeros',
                          name='Hidden_layer_1'))

nn_model.add(layers.Dense(K,
                          activation=hidden_activation_function,
                          use_bias=True, 
                          kernel_initializer='random_normal',
                          bias_initializer='zeros',
                          name='Hidden_layer_2'))

nn_model.add(layers.Dense(K,
                          activation=hidden_activation_function,
                          use_bias=True, 
                          kernel_initializer='random_normal',
                          bias_initializer='zeros',
                          name='Hidden_layer_3'))

nn_model.add(layers.Dense(10, activation=output_activation_function, name='Output_layer'))
nn_model.summary()

optimizer = tf.keras.optimizers.Adam()

loss_function = tf.keras.losses.CategoricalCrossentropy()

metric = ['accuracy']

nn_model.compile(optimizer=optimizer, loss=loss_function, metrics=metric)
nn_model.fit(x=training_images, y=training_labels, 
             batch_size=Nbatch, 
             epochs=EPOCHS, 
             validation_split=validation_split_ratio,
             verbose=1)

print('Accuracy of the model for test-set')
nn_model.evaluate(x=test_images, y=test_labels, verbose=1)

































