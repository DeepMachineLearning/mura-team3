# Import all required modules
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

#from keras import backend as K
#from keras import optimizers
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D,MaxPooling2D,Dropout,Flatten,BatchNormalization,Activation,InputLayer
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

# Parameters
kernel_size=3
conv_kernels=32
drop_prop=0.25

pool_size=2
dense_size=512

batch_size=32
epochs=2

# Prepare data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

n_train, height, width = x_train.shape
depth = 1 # grayscale images
n_test = x_test.shape[0]
n_classes = np.unique(y_train).shape[0]
input_shape=(height,width,depth)


# Normalize data to [0, 1] range
x_train = x_train.astype('float64') 
x_train /= np.max(x_train)
x_train = x_train.reshape(n_train,height,width,depth)

x_test = x_test.astype('float64')
x_test /= np.max(x_test)
x_test = x_test.reshape(n_test,height,width,depth)

y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)

# Data augmentation
train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

def create_cnn_model(): 
	cnn = Sequential()
	
	# Add input layer
	cnn.add(InputLayer(input_shape=input_shape))
	
	# Normalization
	cnn.add(BatchNormalization())
	
	# Conv + Maxpooling
	cnn.add(Conv2D(conv_kernels, (kernel_size, kernel_size), padding="same",activation="relu"))
	cnn.add(MaxPooling2D((pool_size,pool_size)))
	# Dropout
	cnn.add(Dropout(drop_prop))
	
	cnn.add(Conv2D(conv_kernels*2, (kernel_size, kernel_size)))
	cnn.add(MaxPooling2D((pool_size,pool_size)))
	
	# Dropout
	cnn.add(Dropout(drop_prop*2))
	
	cnn.add(Conv2D(conv_kernels*3, (kernel_size, kernel_size)))
	cnn.add(MaxPooling2D((pool_size,pool_size)))
	
	cnn.add(Flatten())
	cnn.add(Dense(dense_size, activation='relu'))
	cnn.add(Dropout(drop_prop))
	cnn.add(Dense(dense_size//2, activation='relu'))
	cnn.add(Dropout(drop_prop))
	cnn.add(Dense(n_classes, activation='softmax'))
	cnn.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer="Adam")
	return cnn

# Create a model
model=create_cnn_model()
# Print the model summary
model.summary()

history = model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=32), steps_per_epoch=len(x_train) / 32, epochs=epochs)

test_loss, test_accuracy = model.evaluate_generator(test_datagen.flow(x_test, Y_test,batch_size=32), verbose=0)
print ('test_loss:%2.2f,test_accuracy:%2.2f' % (test_loss,test_accuracy))

#Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
 
# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
plt.show()
