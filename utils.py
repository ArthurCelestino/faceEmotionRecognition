
from keras.models import Sequential  
from keras.layers.convolutional import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers.core import Activation, Flatten, Dense, Dropout

def create_model(input_shape):
    model = Sequential()

    model.add(Conv2D(128,(3,3),padding = "same", activation = "relu",  input_shape=input_shape))    
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
       
    model.add(Conv2D(64, (3,3), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.1))
    
    model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation("relu"))

    # classificador softmax
    model.add(Dense(7, activation="softmax"))
    
    return model