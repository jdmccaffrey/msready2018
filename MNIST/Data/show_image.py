# show_image.py

import numpy as np
import matplotlib.pyplot as plt

# data file looks like:
# 5 ** 0 .. 23 157 .. 0
# 4 ** 0 .. 255 16 .. 0
# note dummy separator at [1]

def display(txt_file, idx):
  # values between 0-255
  # data file has 1 + 1 + 784 = 786 vals per line, [0] to [785]

  y_data = np.loadtxt(txt_file, delimiter = " ",
    usecols=[0], dtype=np.float32)
  x_data = np.loadtxt(txt_file, delimiter = " ",
    usecols=range(2,786), dtype=np.float32)

  label = int(y_data[idx])  # like '5'
  print("digit = ", str(label), "\n")

  pixels = np.array(x_data[idx,], dtype=np.int)  # to int
  pixels = pixels.reshape((28,28))
  for i in range(28):
    for j in range(28):
      print("%.2X" % pixels[i,j], end="")
      print(" ", end="")
    print("")

  img = np.array(x_data[idx,])   # as float32
  img = img.reshape((28,28))
  plt.imshow(img, cmap=plt.get_cmap('gray_r'))
  plt.show()  

def main():
  print("\nBegin show MNIST image demo \n")

  img_file = ".\\mnist_train_keras_1000.txt"
  display(img_file, idx=0)  # first train image is a '5'

  print("\nEnd \n")

if __name__ == "__main__":
  main()

# =============================================================


  # read everything approach (not preferred)
  #
  # cols = [i for i in range(795) if i != 10]  # skip the '*' in col [10]
  # data = np.loadtxt(img_txt_file, delimiter = " ", usecols=cols, dtype=np.int)
  # data matrix now has n rows with: 10 vals, followed by 784 vals
  # [0] to [9] is label, [10] to [793]
  # label = np.argmax(data[idx,0:10])
  # print("digit = ", str(label), "\n")
  # pixels = np.array(data[idx,10:794], dtype=np.int)
  # pixels = pixels.reshape((28,28))
  # for i in range(28):
  #   for j in range(28):
  #     print("%.2X" % pixels[i,j], end="")
  #     print(" ", end="")
  #   print("")
  # img = np.array(data[idx,10:794], dtype=np.float32)
  # img = img.reshape((28,28))
  # plt.imshow(img, cmap=plt.get_cmap('gray_r'))
  # plt.show()
