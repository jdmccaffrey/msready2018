# imdb_lstm.py
# LSTM for sentiment analysis on the IMDB dataset
# custom load data, no keras.datasets cheating!

import numpy as np
import keras as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# data looks like:
# 0 0 . . 1324 821 . . 0
#
# leading 0s are padding, then index values starting at 4 where
# the index is the frequency, so 4 = 'the', 5 = 'and', etc.
# the last value is 0 (negative review) or 1 (positive)

# ==================================================================

def main():

  print("\nBegin IMDB reviews sentiment demo using Keras LSTM \n") 

  np.random.seed(0)
  max_review_len = 50  # reviews will be padded up to exactly this val
  print("Loading train and test data, max len = %d \n" % max_review_len)

  X_train = np.loadtxt(".\\Data\\imdb_train_50w.txt", delimiter=" ",
    usecols=range(0,max_review_len), dtype=np.float32)
  y_train = np.loadtxt(".\\Data\\imdb_train_50w.txt", delimiter=" ",
    usecols=[max_review_len], dtype=np.float32)
  print(X_train.shape)

  X_test = np.loadtxt(".\\Data\\imdb_test_50w.txt", delimiter=" ",
    usecols=range(0,max_review_len), dtype=np.float32)
  y_test = np.loadtxt(".\\Data\\imdb_test_50w.txt", delimiter=" ",
    usecols=max_review_len, dtype=np.float32)
  print(X_test.shape)

  nw = 129892  # 129888 + 4

  print("\nCreating LSTM model: ")
  embed_vector_len = 32 
  init = K.initializers.glorot_uniform(seed=1)

  model = K.models.Sequential()
  model.add(K.layers.embeddings.Embedding(nw, embed_vector_len,
   mask_zero=True, embeddings_initializer=init,
   input_length=max_review_len))
  model.add(K.layers.LSTM(units=100, dropout=0.2,
    recurrent_initializer=init))  # 100 memory
  model.add(K.layers.Dense(1, activation='sigmoid',
    kernel_initializer=init))

  model.compile(loss='binary_crossentropy', optimizer='adam',
    metrics=['accuracy'])
  print(model.summary())

  print("\nStaring training \n")
  model.fit(X_train, y_train, epochs=5, batch_size=10) 
  print("\nTraining complete")

  loss_acc = model.evaluate(X_test, y_test, verbose=0)
  print("Accuracy on test items = %.2f%%" % (loss_acc[1]*100))

  # mp = ".\\Models\\imdb_model.h5"
  # model.save(mp)

  # mp = ".\\Models\\imdb_model.h5"
  # model = K.models.load_model(mp)

  print("\nMaking a new review")
  # review = "i wish i could say i liked this movie but i can't"
  review = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0,
                      12, 626, 12, 97, 133, 12, 407, 13, 20, 21,
                      12, 186]], dtype=np.float32)
  predict = model.predict(review)
  print("Prediction = ")
  print(predict)

  print("\nEnd IMDB demo \n")

if __name__ == "__main__":
  main()

