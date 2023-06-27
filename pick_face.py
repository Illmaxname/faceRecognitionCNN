#before running this code comment the second returning parameter in readAllImg fuction
import os
import cv2
import time
from read_img import readAllImg

sourcePath = r'.\buffer\!val'
objectPath = r'.\buffer\!val\cropped'

def readPicSaveFace(sourcePath,objectPath,*suffix):
    if not os.path.exists(objectPath):
            os.mkdir(objectPath)
    try:
        resultArray = readAllImg(sourcePath, *suffix)

        count = 1
        face_cascade = cv2.CascadeClassifier('D:\Programs\python3.8\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
        for img in resultArray:
            if type(img) != str:
                if img.ndim == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img
            else:
                continue
            faces = face_cascade.detectMultiScale(gray,1.1,5)
            for (x, y, w, h) in faces:
                listStr = [str(int(time.time())), str(count)]#using time to name new file
                fileName = ''.join(listStr)
                f = cv2.resize(gray[y:(y + h), x:(x + w)], (128, 128))
                cv2.imwrite(objectPath+os.sep+'%s.jpg' % fileName, f)
                count += 1

    except IOError:
        print ("Error")

    else:
        print ('Already read '+str(count-1)+' Faces to Destination '+objectPath)

def convertToGrayscaleAndResize(input_path, output_path):
    # Read the input image
    image = cv2.imread(input_path)

    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to 128x128 pixels
    resized_image = cv2.resize(grayscale_image, (128, 128))

    # Save the modified image
    cv2.imwrite(output_path, resized_image)

if __name__ == '__main__':
    #readPicSaveFace(sourcePath,objectPath,'.jpg')

    input_path = r'.\trShCropped - Copy\d2\e1029921.jpg'
    output_path = r'.\trShCropped - Copy\d2\e1029921.jpg'
    convertToGrayscaleAndResize(input_path, output_path)

