import os
import cv2
import time
from read_img import readAllImg

sourcePath = '.\\training\\8'
objectPath = '.\\trShCropped\\t6'

#Прочитайте все изображения из исходного пути и поместите их в список, затем проверьте их по одному,
# закрепите в них лица и сохраните их в целевом пути
def readPicSaveFace(sourcePath,objectPath,*suffix):
    if not os.path.exists(objectPath):
            os.mkdir(objectPath)
    try:
        #Читать фото, обратите внимание, что первый элемент это имя файла
        resultArray = readAllImg(sourcePath, *suffix)

        # Проверить картинки в списке одну за другой, узнать лица и записать их в целевую папку
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
                listStr = [str(int(time.time())), str(count)]  #Используйте метку времени и порядок чтения в качестве имени файла
                fileName = ''.join(listStr)
                f = cv2.resize(gray[y:(y + h), x:(x + w)], (128, 128))
                cv2.imwrite(objectPath+os.sep+'%s.jpg' % fileName, f)
                count += 1

    except IOError:
        print ("Error")

    else:
        print ('Already read '+str(count-1)+' Faces to Destination '+objectPath)

if __name__ == '__main__':
    readPicSaveFace(sourcePath,objectPath,'.jpg')

