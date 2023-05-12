import cv2
from train_model import Model
from read_data import read_name_list

class Camera_reader(object):
    # Построить модель при инициализации камеры и загрузить обученную модель
    def __init__(self):
        self.model = Model()
        self.model.load()
        self.img_size = 128

    def build_camera(self):
        face_cascade = cv2.CascadeClassifier('D:\Programs\python3.8\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml')
        # Прочитайте имя подпапки в наборе данных набора данных
        name_list = read_name_list('.\\training')

        # Откройте камеру и начните читать экран
        cameraCapture = cv2.VideoCapture(0)
        success, frame = cameraCapture.read()

        while success and cv2.waitKey(1) == -1:
             success, frame = cameraCapture.read()
             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # изображение становится серым
             faces = face_cascade.detectMultiScale(gray, 1.3, 5) #Распознавать лица
             for (x, y, w, h) in faces:
                 ROI = gray[x:x + w, y:y + h]
                 ROI = cv2.resize(ROI, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                 label,prob = self.model.predict(ROI)  #Используем модель для сравнения лиц, распознаваемых cv2
                 if prob >0.7:    #Если модель считает, что вероятность выше 70%,
                     # она будет отображаться как существующая метка в модели.
                     show_name = name_list[label]
                 else:
                     show_name = 'Stranger'
                 cv2.putText(frame, show_name, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)  # показать имя
                 frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Рисуем квадрат в области лица
             cv2.imshow("Camera", frame)

        cameraCapture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    camera = Camera_reader()
    camera.build_camera()

