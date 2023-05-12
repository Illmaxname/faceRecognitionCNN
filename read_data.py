import os
import cv2
import numpy as np

from read_img import endwith

# Введите путь к файлу, прочитайте изображения в каждой папке под ним и присвойте каждой папке разные метки.
#Вернуть список img, вернуть список, соответствующий метке, и вернуть количество папок (меток несколько)

def read_file(path):
    img_list = []
    label_list = []
    dir_counter = 0
    IMG_SIZE = 128

    # Прочитать все файлы jpg во всех подпапках по пути и сохранить их в списке

    for child_dir in os.listdir(path):
         child_path = os.path.join(path, child_dir)

         for dir_image in  os.listdir(child_path):
             if endwith(dir_image,'jpg'):
                img = cv2.imread(os.path.join(child_path, dir_image))
                resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                recolored_img = cv2.cvtColor(resized_img,cv2.COLOR_BGR2GRAY)
                img_list.append(recolored_img)
                label_list.append(dir_counter)

         dir_counter += 1

    # Возвращенный img_list преобразуется в формат np.array
    img_list = np.array(img_list)

    return img_list, label_list, dir_counter

# Прочитайте папку набора обучающих данных и верните их имена в список

def read_name_list(path):
    name_list = []
    for child_dir in os.listdir(path):
        name_list.append(child_dir)
    return name_list

if __name__ == '__main__':
    img_list, label_lsit, counter = read_file('.\\trainingShort')
    print(counter)

