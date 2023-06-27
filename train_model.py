from dataSet import DataSet
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
import numpy as np

#Building CNN model face recognition
class Model(object):
    FILE_PATH = r".\model.h5"  #Path to model
    IMAGE_SIZE = 128 #Defining size of images: 128x128 pixels

    def __init__(self):
        self.model = None

    def read_trainData(self, dataset):
        self.dataset = dataset

    # Building CNN model
    def build_model(self):
        self.model = Sequential()
        self.model.add(
            Convolution2D(
                filters=16,
                kernel_size=(5, 5),
                padding='same',
                data_format='channels_last',
                input_shape=self.dataset.X_train.shape[1:]
            )
        )

        self.model.add(Activation('relu'))
        self.model.add(
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='same'
            )
        )

        self.model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        self.model.add(Convolution2D(filters=64, kernel_size=(5, 5), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))


        self.model.add(Flatten())
        self.model.add(Dense(1024))
        self.model.add(Activation('relu'))
        #self.model.add(Dropout(0.1))  # Dropout rate(10% of the units will be dropped during training)
        self.model.add(Dense(self.dataset.num_classes))
        self.model.add(Activation('softmax'))
        self.model.summary()

    def train_model(self):
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        batch_size = 100
        epochs = 13

        self.model.fit(
            self.dataset.X_train,
            self.dataset.Y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.dataset.X_test, self.dataset.Y_test),
        )

    def evaluate_model(self):
        print('\nTesting---------------')
        loss, accuracy = self.model.evaluate(self.dataset.X_test, self.dataset.Y_test)

        print('test loss;', loss)
        print('test accuracy:', accuracy)

    def save(self, file_path=FILE_PATH):
        print('Model Saved.')
        self.model.save(file_path)

    def load(self, file_path=FILE_PATH):
        print('Model Loaded.')
        self.model = load_model(file_path)

    def predict(self, img):
        img = img.reshape((1, self.IMAGE_SIZE, self.IMAGE_SIZE, 1))
        img = img.astype('float32')
        img = img / 255.0

        result = self.model.predict(img)  #Probability that the image belongs to the label's prototype
        max_index = np.argmax(result)  #Label with max probability

        return max_index, result[0][max_index]  #The first parameter is the index of the label with the highest probability
        # and the second parameter is the corresponding probability

if __name__ == '__main__':
    dataset = DataSet(r'.\trShCropped - Copy')
    dataset.check()
    model = Model()
    model.read_trainData(dataset)
    model.build_model()
    model.train_model()
    model.evaluate_model()
    model.save()

