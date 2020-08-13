import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.models import Sequential
from numpy import reshape
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

dataset = keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = dataset.load_data()

x_train = x_train /255.0
x_test = x_test/255.0

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1], x_test.shape[2], 1)

y_test = to_categorical(y_test)
y_train = to_categorical(y_train)

datagen = ImageDataGenerator(width_shift_range=0.1,
                             horizontal_flip=True)

it_train = datagen.flow(x_train, y_train, batch_size=60)
steps = int(x_train.shape[0]/60)



class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(x_train[44].shape)

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', input_shape=(28,28,1)))
model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))

print(model.summary())

opt = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

filepath = 'BEST-CNN-{loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callback_list = [checkpoint]

model.fit_generator(it_train, steps_per_epoch=steps, epochs=10, callbacks=callback_list, validation_data=(x_test, y_test), verbose=0)


_, acc = model.evaluate(x_test, y_test, verbose=0)

print('acc: %.3f' % (acc*100))

