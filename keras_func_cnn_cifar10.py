import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Dense, Input, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import time


start_time = time.time()


print('[Start][cnn][cifar10] Set hyper parameter -----')
# Hyper parameter
#  - batch_size : number of training data per batch learning
#  - num_classes: number of classes of label data
#  - epochs     : number of learning cycle
batch_size = 128
num_classes = 10
epochs = 20


print('[Start][cnn][cifar10] Data download -----')
# cifar10 data :shape[0]:shape[1]:shape[2]: shape[3]: Remarks
# x_train      :  50000 :     32 :     32 :       3 : Data for learning
# y_train      :  50000 :     -- :     -- :      -- : Label data for learning data
# x_test       :  10000 :     32 :     32 :       3 : Data for testing
# y_test       :  10000 :     -- :     -- :      -- : Label data for testing data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


print('[Start][cnn][cifar10] Display first 10 data for learning -----')
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.title("M_%d" % i)
    plt.axis("off")
    plt.imshow(x_train[i].reshape(32, 32, 3), cmap=None)
plt.show()


print('[Start][cnn][cifar10] Reshape learning data -----')
# Change the type of x_train/x_test to 'float32'
# Data set data are represented as 0-255.
# For normalizing these, divide by 255, these become the values of 0.0-1.0.
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
# Data set(cifar10) are represented as RGB.
# No need to add the dimension as channel.


print('[Start][cnn][cifar10] Reshape label data -----')
# Using to_categorical method, each value of label data is changed to class matrix.
y_train = keras.utils.to_categorical(y_train, num_classes)
# 50000 -> 50000, 10(=num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# 10000 -> 10000, 10(=num_classes)


print('[Start][cnn][cifar10] Construct neural network -----')
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# Tuple af input is the size of vertical, horizontal and channel.
input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
print('x_test:', x_test.shape)
inputs = Input(shape=input_shape)  # 32*32*3
x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(inputs)
x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(x)
x = MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid')(x)
x = Dropout(rate=0.25)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(rate=0.5)(x)
outputs = Dense(10, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=Adadelta(),
              metrics=['accuracy'])


start_learning_time = time.time()


print('[Start][cnn][cifar10] Learn using training data -----')
es = EarlyStopping(monitor='val_loss', patience=2)
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=[es])


start_evaluating_time = time.time()


print('[Start][cnn][cifar10] Evaluate using label data -----')
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


end_time = time.time()


print('[Start][cnn][cifar10] Display the result of learning -----')
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = len(loss)
plt.plot(range(epochs), loss, marker='.', label='loss(training data)')
plt.plot(range(epochs), val_loss, marker='.', label='val_loss(evaluation data')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
timestamp = str(int(time.time()))
filename = 'mygraph2_' + timestamp + '.png'
plt.savefig(filename)  # Save the graph as file
plt.show()             # Display the graph


print('[Start][cnn][cifar10] Save the model and its weight -----')
open('cifar10_cnn_' + timestamp + '.json', "w").write(model.to_json())  # Save the model as json
model.save_weights('cifar10_cnn_' + timestamp + '.h5')             # Save the weight of model as h5 format


print('[Start][cnn][cifar10] Display the result of time measurement -----')
print('start_time:', start_time)
print('start_learning_time:', start_learning_time)
print('start_evaluating_time:', start_evaluating_time)
print('end_time:', end_time)
print('elapsed time(Prep):', start_learning_time-start_time)
print('elapsed time(Learn):', start_evaluating_time-start_learning_time)
print('elapsed time(Eval):', end_time-start_evaluating_time)
print('elapsed time(Total):', end_time-start_time)


