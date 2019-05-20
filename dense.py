from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D

# use Matplotlib (don't ask)
import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
import pandas as pd
import json

# this is the size of our encoded representations
encoding_dim = 64 # 32 floats -> compression of factor 24.5, assuming the input is 37800 floats

input_img = Input(shape=(37800,))
encoded = Dense(11200, activation='relu')(input_img)
encoded = Dense(512, activation='relu')(encoded)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(512, activation='relu')(decoded)
decoded = Dense(11200, activation='relu')(decoded)
decoded = Dense(37800, activation='sigmoid')(decoded)


autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)



test_data = pd.read_csv('test.tsv', sep= '\t')
rgb_vector = test_data['rgb_vector']

arr = []

for vec in rgb_vector:
    vec = json.loads(vec)
    arr.append(vec)

arr = np.array(arr)

x_train, x_test = arr[:200,:], arr[200:,:]

autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=100,
                shuffle=True,
                validation_data=(x_test, x_test))

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(150, 84,3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display encoded images
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(encoded_imgs[i].reshape(8,8))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_imgs[i].reshape(150,84,3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('plot.png')
