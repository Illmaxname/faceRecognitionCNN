from read_data import read_name_list, read_file
from train_model import Model
import cv2
import os

def test_onePicture(path):
    model = Model()
    model.load()
    img = cv2.imread(path)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    picType, prob = model.predict(img)
    if picType != -1:
        name_list = read_name_list('.\\trShCropped')
        print(name_list[picType], prob)
    else:
        print("Don't know this person")

#Reading all images in folder & subfolders
def test_onBatch(path):
    model = Model()
    model.load()

    index = 0
    img_list, label_lsit, counter = read_file(path)
    for img in img_list:
        picType, prob = model.predict(img)
        if picType != -1:
            index += 1
            name_list = read_name_list(r'.\trShCropped')
            print(name_list[picType])
        else:
            print(" Don't know this person")

    return index

def test_onBatch2(path):
    model = Model()
    model.load()
    index = 0
    name_list = read_name_list('.\\trShCropped')

    # Iterate through the images in the specified folder
    for filename in os.listdir(path):
        img_path = os.path.join(path, filename)
        img = cv2.imread(img_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        face_cascade = cv2.CascadeClassifier('D:\Programs\python3.8\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Extract the region of interest (ROI) containing the face
            ROI = gray[y:y + h, x:x + w]
            ROI = cv2.resize(ROI, (model.IMAGE_SIZE, model.IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

            # Perform classification using the model
            label, prob = model.predict(ROI)

            if prob > 0.7:
                # If the probability is above 70%, display the recognized name
                show_name = name_list[label]
            else:
                show_name = 'Stranger'

            print(show_name)
            index += 1

    return index

if __name__ == '__main__':
    #test_onePicture('.\\test\\1\\andrey.jpg')
    #test_onePicture('.\\test\\2\\daniil.jpg')
    #test_onePicture(r'.\photos\me.jpg')
    #test_onePicture('.\\test\\4\\photo.jpg')#serg
    #test_onePicture('.\\test\\5\\tamara.jpg')
    #test_onePicture('.\\test\\6\\photo.jpg')#ul
    #test_onePicture('.\\test\\7\\photo.jpg')#yu
    test_onBatch(r'.\test')


