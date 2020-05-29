from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.datasets import mnist
from keras.utils import np_utils
import keras

(X_train, y_train), (X_test, y_test)  = mnist.load_data()



X_train = X_train.reshape( X_train.shape[0] , X_train[0].shape[0] , X_train[1].shape[0] , 1 )
X_test = X_test.reshape( X_test.shape[0] , X_test[0].shape[0] , X_test[1].shape[0] , 1 )



input_shape = ( X_train[0].shape[0] , X_train[1].shape[0], 1)




X_train = X_train.astype('float32')
X_test = X_test.astype('float32')



X_train /= 255
X_test /= 255


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]
num_pixels = X_train.shape[1] * X_train.shape[2]



def CRP_layer(layers , filters , filterSize , poolSize , strideSize):
  for i in range(layers):
    model.add(Conv2D(filters, (filterSize, filterSize),
                 padding = "same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (poolSize, poolSize), strides = (strideSize, strideSize)))
  return


model = Sequential()

model.add(Conv2D(20, (5, 5),
                 padding = "same", 
                 input_shape = input_shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))


model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))


model.add(Dense(num_classes))
model.add(Activation("softmax"))
           
model.compile(loss = 'categorical_crossentropy',
              optimizer = keras.optimizers.Adadelta(),
              metrics = ['accuracy'])


batch_size = 128
epochs = 3

history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test),
          shuffle=True)

scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

oldaccuracy = open('accuracy.txt','r')
old = float(oldaccuracy.read())
oldaccuracy.close()
new  = scores[1]

if new > old :
  accuracyStored = open('accuracy.txt','w')
  accuracyStored.write(str(new))
  accuracyStored.close()
  model.save("ArpitFinalModel.h5")
else :
  print("No improverment , model is saved with previous accuracy of {}".format(old))