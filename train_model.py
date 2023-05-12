from dataSet import DataSet
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
import numpy as np

# Построить модель распознавания лиц на основе CNN
class Model(object):
    FILE_PATH = ".\\model.h5"  # Где модель хранится и читается
    IMAGE_SIZE = 128  # Изображение лица, принятое моделью, должно быть 128*128.

    def __init__(self):
        self.model = None

    # Чтение экземпляра класса DataSet в качестве источника данных для обучения
    def read_trainData(self, dataset):
        self.dataset = dataset

    # Построить модель CNN, один слой свертки, один слой объединения, один слой свертки, один слой объединения,
    # полную связь после сглаживания и, наконец, классификацию
    def build_model(self):
        self.model = Sequential()
        self.model.add(
            Convolution2D(
                filters=32,
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

        self.model.add(Convolution2D(filters=64, kernel_size=(5, 5), padding='same',
                data_format='channels_last',
                input_shape=self.dataset.X_train.shape[1:]))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        self.model.add(Convolution2D(filters=128, kernel_size=(5, 5), padding='same',
                data_format='channels_last',
                input_shape=self.dataset.X_train.shape[1:]))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        self.model.add(Flatten())
        self.model.add(Dense(1024))
        self.model.add(Activation('relu'))

        self.model.add(Dense(self.dataset.num_classes))
        self.model.add(Activation('softmax'))
        self.model.summary()


    # Функция обучения модели, конкретный оптимизатор и потеря могут быть выбраны по-разному
    def train_model(self):
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        # epochs, batch_size — настраиваемые параметры, epochs — количество раундов обучения,
        # batch_size — сколько выборок обучается каждый раз
        #self.model.fit(self.dataset.X_train, self.dataset.Y_train, epochs=2, batch_size=10)

        batch_size = 8
        epochs = 16

        self.model.fit(
            self.dataset.X_train,
            self.dataset.Y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.dataset.X_test, self.dataset.Y_test),
            #shuffle=True
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

    # Необходимо убедиться, что входной img является изображением лица после поседения (канал = 1) и размером IMAGE_SIZE
    def predict(self, img):
        img = img.reshape((1, self.IMAGE_SIZE, self.IMAGE_SIZE, 1))
        img = img.astype('float32')
        img = img / 255.0

        result = self.model.predict(img)  # Рассчитать вероятность того, что изображение принадлежит определенному ярлыку
        max_index = np.argmax(result)  # найти наибольшую вероятность

        return max_index, result[0][max_index]  # Первый параметр - это индекс метки с наибольшей вероятностью, а второй параметр - соответствующая вероятность

if __name__ == '__main__':
    dataset = DataSet('.\\trainingShort')
    dataset.check()
    model = Model()
    model.read_trainData(dataset)
    model.build_model()
    model.train_model()
    model.evaluate_model()
    model.save()

