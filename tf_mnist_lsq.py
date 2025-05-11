import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)

(training_images, training_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
depth = 10
training_labels = tf.one_hot(training_labels, depth)
test_labels = tf.one_hot(test_labels, depth)
print('sample labels from training set:{}'.format(training_labels[1:6]))

print('Training Images Dataset Shape: {}'.format(training_images.shape))
print('No. of Training Images Dataset Labels: {}'.format(len(training_labels)))
print('Test Images Dataset Shape: {}'.format(test_images.shape))
print('No. of Test Images Dataset Labels: {}'.format(len(test_labels)))

K = 100    # K is the number of nodes in the first hidden layer
dt = 0.002    # dt is learning rate
N_training_set = 60000
validation_split_ratio = 0.2
Nbatch = 10
EPOCHS = 20
M_step = EPOCHS * N_training_set * (1 - validation_split_ratio) / Nbatch
T = M_step * dt
print('Number of SGD steps {}'.format(M_step))

xmean = np.mean(training_images)    # training set of size 60000 * 28 * 28
xstd = np.sqrt(np.var(training_images))
training_images = (training_images-xmean)/xstd    #Normalization of training_data
test_images = (test_images-xmean)/xstd    #Normalization of test_data

tf.random.set_seed(123)

input_data_shape = (28, 28)
hidden_activation_function = tf.keras.activations.sigmoid
nn_model = keras.Sequential()
nn_model.add(layers.Flatten(input_shape=input_data_shape, name='Input_layer'))    # Flatten maps a tensor directly to a vector
nn_model.add(layers.Dense(K,
                          activation=hidden_activation_function,
                          use_bias=True,
                          kernel_initializer='random_normal',
                          bias_initializer='zeros',
                          name='Hidden_layer_1'))
nn_model.add(layers.Dense(10, use_bias=False, kernel_initializer='random_normal', name='Output_layer'))
nn_model.summary()

optimizer = tf.keras.optimizers.SGD(learning_rate=dt)


loss_function = tf.keras.losses.MeanSquaredError()

metric = ['accuracy']

nn_model.compile(optimizer=optimizer, loss=loss_function, metrics=metric)
nn_model.fit(x=training_images, y=training_labels, 
             batch_size=Nbatch, 
             epochs=EPOCHS, 
             validation_split=validation_split_ratio,
             verbose=1)

print('Accuracy of the model for test-set')
nn_model.evaluate(x=test_images, y=test_labels, verbose=1)

































