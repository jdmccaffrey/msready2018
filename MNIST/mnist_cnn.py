# mnist_cnn_2.py
# Keras 2.1.5 TensorFlow 1.7.0
# Anaconda3 4.1.1 (Python 3.5.2)

# assumes data looks like:
# 5 ** 0 0 152 27 .. 0
# 3 ** 0 0 38 122 .. 0
# single label val at [0] and 784 vals at [2-785] = [2-786)
# dummy ** seperator at [1] 

import numpy as np
import keras as K

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def encode_y(y_mat, y_dim):
  n = len(y_mat)  # rows
  result = np.zeros(shape=(n, y_dim), dtype=np.float32)
  for i in range(n):  # each row
    val = int(y_mat[i])    # like 5
    result[i][val] = 1
  return result

def main():
  print("\nMNIST CNN demo using Keras/TensorFlow ")
  np.random.seed(1)

  print("Loading MNIST train and test data into memory \n")
  train_file = ".\\Data\\mnist_train_keras_1000.txt"
  test_file = ".\\Data\\mnist_test_keras_100.txt"

  train_x = np.loadtxt(train_file, usecols=range(2,786),
    delimiter=" ",  skiprows=0, dtype=np.float32)
  train_y = np.loadtxt(train_file, usecols=[0],
    delimiter=" ", skiprows=0, dtype=np.float32)

  train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
  train_x /= 255
  train_y = encode_y(train_y, 10)  # one-hot

  test_x = np.loadtxt(test_file, usecols=range(2,786),
    delimiter=" ",  skiprows=0, dtype=np.float32)
  test_y = np.loadtxt(test_file, usecols=[0],
    delimiter=" ", skiprows=0, dtype=np.float32)

  test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)
  test_x /= 255 
  test_y = encode_y(test_y, 10)  # one-hot 

  input_dim = (28, 28, 1)
  output_dim = 10
  batch_size = 128
  max_epochs = 12  # 12

  init = K.initializers.RandomNormal(mean=0.0, stddev=0.10, seed=1)
  model = K.models.Sequential()
  model.add(K.layers.Conv2D(filters=32, kernel_size=(3,3),
    strides=(1,1), activation='relu', input_shape=input_dim,
    kernel_initializer=init))
  model.add(K.layers.Conv2D(filters=64, kernel_size=(3,3),
    strides=(1,1), activation='relu', kernel_initializer=init))
  model.add(K.layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(K.layers.Dropout(0.25))
  model.add(K.layers.Flatten())
  model.add(K.layers.Dense(units=100, activation='relu',
    kernel_initializer=init))
  model.add(K.layers.Dropout(rate=0.5))
  model.add(K.layers.Dense(output_dim, activation='softmax',
    kernel_initializer=init))

  model.compile(loss='categorical_crossentropy',
    optimizer='adadelta', metrics=['accuracy'])

  print("Starting training ")
  model.fit(train_x, train_y, batch_size=batch_size,
    epochs=max_epochs, verbose=1,
    validation_data=(train_x, train_y))
  print("Training complete")

  loss_acc = model.evaluate(test_x, test_y, verbose=0)
  print("\nTest data loss/error = %0.4f" % loss_acc[0])
  print("Test data accuracy   = %0.2f%%" % (loss_acc[1]*100))

  # make a prediction on nonsense digit
  print("\nMaking prediction for a nonsense digit")
  input_vec = np.zeros(shape=(1,28,28,1), dtype=np.float32)
  for i in range(5,23): input_vec[0,i,i,0] = 254
  for i in range(10,15): input_vec[0,i,20-i,0] = 200
  pred_probs = model.predict(input_vec)
  pred_digit = np.argmax(pred_probs)
  print("Prediction: ", pred_digit)
  

  print("\nEnd MNIST demo \n")

if __name__=="__main__":
  main()
