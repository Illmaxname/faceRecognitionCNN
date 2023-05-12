'''Этот код читает все изображения в заданном каталоге, которые имеют заданные расширения файлов,
и выводит первое изображение в окне с помощью библиотеки OpenCV.

Конкретно, функция `readAllImg(path, *suffix)` принимает аргумент `path`, представляющий путь к каталогу,
который нужно прочитать, и `*suffix`, переменное количество дополнительных аргументов, которые задают расширения файлов,
 которые нужно прочитать (в данном случае только ".jpg"). Функция возвращает список,
 содержащий имена всех прочитанных файлов и массив самих изображений.

Функция `endwith(s, *endstring)` проверяет, заканчивается ли строка `s` хотя бы одним из строковых аргументов `*endstring`.

Далее, если скрипт запущен как основная программа, то он вызывает функцию `readAllImg` с аргументами
".\\training" (текущий каталог и подкаталог "training") и ".jpg" (файлы с расширением ".jpg").
Затем он выводит имя первого изображения в списке, создает окно "Image" и показывает в нем первое изображение в списке.
Окно с изображением остается открытым до тех пор, пока пользователь не закроет его.'''

import os
import cv2

def readAllImg(path, *suffix):
    try:
        s = os.listdir(path)
        resultArray = []
        fileName = os.path.basename(path)
        resultArray.append(fileName)

        for i in s:
            if endwith(i, suffix):
                document = os.path.join(path, i)
                img = cv2.imread(document)
                resultArray.append(img)

    except IOError:
        print("Error")

    else:
        print("Done")
        return resultArray

def endwith(s, *endstring):
   resultArray = map(s.endswith, endstring)
   if True in resultArray:
       return True
   else:
       return False

if __name__ == '__main__':

  result = readAllImg(".\\photos", '.jpg')
  print(result[0])
  #cv2.namedWindow("Image")
  #cv2.imshow("Image", result[69])
  #cv2.waitKey(0)
  #cv2.destroyAllWindows()