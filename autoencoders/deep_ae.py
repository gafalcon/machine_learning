from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

# this is the size of our encoded representations
encoding_dim = 32 # 32 floats -> compression of factor 24.5, assuming input is 784 floats

# input placeholder
input_img = Input(shape=(784,))
# encoded representation of the input
encoded1 = Dense(128, activation='relu')(input_img)
encoded2 = Dense(64, activation='relu')(encoded1)
encoded3 = Dense(32, activation='relu')(encoded2)

# decoded is the lossy reconstruction of the input
decoded1 = Dense(64, activation='relu')(encoded3)
decoded2 = Dense(128, activation='relu')(decoded1)
decoded3 = Dense(784, activation='sigmoid')(decoded2)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded3)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded3)

# Decoder Model
# create placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(32,))
# retrieve the last layer of the autoencoder model
decoder_layers = autoencoder.layers[-3:]
# create the decoder model
decoder_layer1 = decoder_layers[0]
decoder_layer2 = decoder_layers[1]
decoder_layer3 = decoder_layers[2]
decoder = Model(encoded_input, decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))

#per-pixel binary crossentropy loss.
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

#Prepare input data.
from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()
#normalize values and flatten
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(x_train.shape)
print(x_test.shape)

#Train autoencoder for 50 epochs
autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# encode and decode some digits
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

#visualize reconstructed inputs
import matplotlib.pyplot as plt

n = 10 # How many digits we will display
plt.figure(figsize=(20,4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
