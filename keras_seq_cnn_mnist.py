import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import time


start_time = time.time()


print('[Start][cnn][mnist] Set hyper parameter -----')
# Hyper parameter
#  - batch_size : number of training data per batch learning
#  - num_classes: number of classes of label data
#  - epochs     : number of learning cycle
batch_size = 128
num_classes = 10
epochs = 20


print('[Start][cnn][mnist] Data download -----')
# mnist data :shape[0]:shape[1]:shape[2]: Remarks
# x_train    :  60000 :     28 :     28 : Data for learning
# y_train    :  60000 :     -- :     -- : Label data for learning data
# x_test     :  10000 :     28 :     28 : Data for testing
# y_test     :  10000 :     -- :     -- : Label data for testing data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

"""
print('[Start][cnn][mnist] Display first 10 data for learning -----')
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.title("M_%d" %i)
    plt.axis("off")
    plt.imshow(x_train[i].reshape(28, 28), cmap=None)
plt.show()
"""

print('[Start][cnn][mnist] Reshape learning data -----')
# Change the type of x_train/x_test to 'float32'
# Data set are represented as 0-255.
# For normalizing these, divide by 255, these become the values of 0.0-1.0.
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
# Data set(mnist) are represented as greyscale.
# Need to add the dimension as channel to tuple.
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
# 60000, 28, 28 -> 60000, 28, 28, 1
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
# 10000, 28, 28 -> 10000, 28, 28, 1


print('[Start][cnn][mnist] Reshape label data -----')
# Using to_categorical method, each value of label data is changed to class matrix.
y_train = keras.utils.to_categorical(y_train, num_classes)
# 60000 -> 60000, 10(=num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# 10000 -> 10000, 10(=num_classes)


print('[Start][cnn][mnist] Construct neural network -----')
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# Tuple af input is the size of vertical, horizontal and channel.
input_shape = (x_train.shape[1], x_train.shape[2], 1)
print('x_test:', x_test.shape)
model = Sequential()
# 1st layer      : input layer(Convolutional layer)
#    filters     :          32 : number of filters(=number of layer's output)
#    kernel_size :      (3, 3) : size of each filter
#    strides     :      (1, 1) : pitch of sliding window
#    padding     :     'valid' : no padding - other option:https://github.com/vdumoulin/conv_arithmetic
#    activation  :      'relu' : activation function
#    input_shape : input_shape : layer's input
model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu', input_shape=input_shape))
# 2nd layer      : hidden layer(Convolutional layer)
#    filters     :          32 : number of filters(=number of layer's output)
#    kernel_size :      (3, 3) : size of each filter
#    strides     :      (1, 1) : pitch of sliding window
#    padding     :     'valid' : no padding - other option:https://github.com/vdumoulin/conv_arithmetic
#    activation  :      'relu' : activation function
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
# 3rd layer      : hidden layer(Pooling layer)
#    pool_size   :      (2, 2) : **size of each filter
#    strides     :      (1, 1) : **pitch of sliding window
#    padding     :     'valid' : no padding - other option:https://github.com/vdumoulin/conv_arithmetic
# model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
# **
model.add(Dropout(rate=0.25))
# **
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=Adadelta(),
              metrics=['accuracy'])


start_learning_time = time.time()


print('[Start][cnn][mnist] Learn using training data -----')
es = EarlyStopping(monitor='val_loss', patience=2)
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=[es])


start_evaluating_time = time.time()


print('[Start][cnn][mnist] Evaluate using label data -----')
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


end_time = time.time()


print('[Start][cnn][mnist] Display the result of learning -----')
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = len(loss)
plt.plot(range(epochs), loss, marker='.', label='loss(training data)')
plt.plot(range(epochs), val_loss, marker='.', label='val_loss(evaluation data')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
filename = 'mygraph_' + str(int(time.time())) + '.png'
plt.savefig(filename)  # Save the graph as file
plt.show()             # Display the graph


print('start_time:', start_time)
print('start_learning_time:', start_learning_time)
print('start_evaluating_time:', start_evaluating_time)
print('end_time:', end_time)


print('elapsed time(Prep):', start_learning_time-start_time)
print('elapsed time(Learn):', start_evaluating_time-start_learning_time)
print('elapsed time(Eval):', end_time-start_evaluating_time)
print('elapsed time(Total):', end_time-start_time)


