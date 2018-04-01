import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
#%matplotlib inline #using only via Jupyter Notebook
import matplotlib.pyplot as plt


print('[Start][dence][mnist] Set hyper parameter -----')
# Hyper parameter
#  - batch_size : number of training data per batch learning
#  - num_classes: number of classes of label data
#  - epochs     : number of learning cycle
batch_size = 128
num_classes = 10
epochs = 20
print('[End][dence][mnist] Set hyper parameter -----')


print('[Start][dence][mnist] Data download -----')
# mnist data :shape[0]:shape[1]:shape[2]: Remarks
# x_train    :  60000 :     28 :     28 : Data for learning
# y_train    :  60000 :     -- :     -- : Label data for learning data
# x_test     :  10000 :     28 :     28 : Data for testing
# y_test     :  10000 :     -- :     -- : Label data for testing data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('[End][dence][mnist] Data download -----')


print('[Start][dence][mnist] Display first 10 data for learning -----')
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.title("M_%d" % i)
    plt.axis("off")
    plt.imshow(x_train[i].reshape(28, 28), cmap=None)
plt.show()
print('[End][dence][mnist] Display first 10 data for learning -----')


print('[Start][dence][mnist] Reshape learning data -----')
x_train = x_train.reshape(60000, 784) # 60000, 28, 28 -> 60000, 784
x_test = x_test.reshape(10000, 784) # 10000, 28, 28 -> 10000, 784
# Change the type of x_train/x_test to 'float32'
# mnist data are represented as 0-255.
# For normalizing these, divide by 255, these become the values of 0.0-1.0.
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('[End][dence][mnist] Reshape learning data -----')


print('[Start][dence][mnist] Reshape label data -----')
# Using to_categorical method, each value is changed to class matrix.
y_train = keras.utils.to_categorical(y_train, num_classes)
# 60000 -> 60000, 10(=num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# 10000 -> 10000, 10(=num_classes)
print('[End][dence][mnist] Reshape label data -----')


print('[Start][dence][mnist] Construct neural network -----')
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
model = Sequential()
# input layer:
#    input dim  :784(28*28)
#    output dim :512
#    activation :relu
#    dropout    :0.2
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
# hidden layer:
#    input dim  :(512)
#    output dim :512
#    activation :relu
#    dropout    : 0.2
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
# output layer
#    input dim  :(512)
#    output dim :10
#    activation :softmax
model.add(Dense(10, activation='softmax'))
# num of params
#   Input layer  : 401920 : (784+1)*512
#   Hidden layer : 262656 : (512+1)*512
#   Output layer :   5130 : (512+1)*10
model.summary()
# loss      : Name of objective function(loss function) such as categorical_crossentropy
# optimizer : Optimization algorithm of gradient descent such as sgd, RMSprop
# metrics   : Function list for evaluating models when training and testing
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
print('[End][dence][mnist] Construct neural network -----')


print('[Start][dence][mnist] Learn using training data -----')
es = EarlyStopping(monitor='val_loss', patience=2)
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=[es])
print('[End][dence][mnist] Learn using training data -----')


print('[Start][dence][mnist] Evaluate using label data -----')
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('[End][dence][mnist] Evaluate using label data -----')


print('[Start][dence][mnist] Display the result of learning -----')
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = len(loss)
plt.plot(range(epochs), loss, marker='.', label='loss(training data)')
plt.plot(range(epochs), val_loss, marker='.', label='val_loss(evaluation data')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
print('[End][dence][mnist] Display the result of learning -----')




