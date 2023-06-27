from read_data import read_file
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import random
import numpy as np
import cv2

#Store and format data class
class DataSet(object):
   def __init__(self, path):
       self.num_classes = None
       self.X_train = None
       self.X_test = None
       self.Y_train = None
       self.Y_test = None
       self.img_size = 128
       self.extract_data(path) #reading training data

   def extract_data(self, path):
        imgs, labels, counter = read_file(path)#reading images, lables and amount of labels
        print(counter)
        test_im, test_lab, test_counter = read_file(r'.\test')

        #Mixing train and validation data
        X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.2, random_state=random.randint(0, 100))

        #Manually creating train and test dataset
        '''X_train = imgs
        y_train = labels
        X_test = test_im
        y_test = test_lab'''

        X_train, y_train = self.augment_data(X_train, y_train)

        resized_X_train = []
        for image in X_train:
            resized_image = cv2.resize(image, (self.img_size, self.img_size))
            resized_X_train.append(resized_image)

        resized_X_test = []
        for image in X_test:
            resized_image = cv2.resize(image, (self.img_size, self.img_size))
            resized_X_test.append(resized_image)

        X_train = np.array(resized_X_train)
        X_test = np.array(resized_X_test)

        #format and normalization data based on tensorflow
        X_train = X_train.reshape(X_train.shape[0], self.img_size, self.img_size, 1) / 255.0
        X_test = X_test.reshape(X_test.shape[0], self.img_size, self.img_size, 1) / 255.0

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        #Convert labels in binary class matrix
        Y_train = np_utils.to_categorical(y_train, num_classes=counter)
        Y_test = np_utils.to_categorical(y_test, num_classes=counter)

        #Assign formated data to classes attributes
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.num_classes = counter

   def augment_data(self, X_train, y_train):
       X_train_augmented = []
       y_train_augmented = []

       for image, label in zip(X_train, y_train):
           # Original image
           X_train_augmented.append(image)
           y_train_augmented.append(label)

           # Flipped image
           flipped_image = np.flip(image, axis=1)
           X_train_augmented.append(flipped_image)
           y_train_augmented.append(label)

           # Random crop
           crop_size = int(np.random.uniform(0.8, 1.0) * image.shape[0])
           cropped_image = image[:crop_size, :crop_size]
           X_train_augmented.append(cropped_image)
           y_train_augmented.append(label)

           # Add noise
           noise = np.random.normal(0, 0.1, image.shape)
           noisy_image = image + noise
           X_train_augmented.append(noisy_image)
           y_train_augmented.append(label)

           # Rotate
           '''rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
           X_train_augmented.append(rotated_image)
           y_train_augmented.append(label)'''

       return np.array(X_train_augmented, dtype=object), np.array(y_train_augmented, dtype=object)


   def check(self):
       print('num of dim:', self.X_test.ndim)
       print('shape:', self.X_test.shape)
       print('size:', self.X_test.size)

       print('num of dim:', self.X_train.ndim)
       print('shape:', self.X_train.shape)
       print('size:', self.X_train.size)

if __name__ == '__main__':
    image = cv2.imread(r'.\trShCropped\m4\16844467922.jpg')
    """flipped_image = np.flip(image, axis=1)
    cv2.imwrite(r'.\photos\flipped_image.jpg', flipped_image)
    crop_size = int(np.random.uniform(0.8, 1.0) * image.shape[0])
    cropped_image = image[:crop_size, :crop_size]
    cv2.imwrite(r'.\photos\cropped_image.jpg', cropped_image)"""
    noise = np.random.normal(0, 5.5, image.shape)
    noisy_image = image + noise
    cv2.imwrite(r'.\photos\noisy_image.jpg', noisy_image)