import cv2
import os

def crop_all_faces(directory_path, scaleFactor=1.001, face_detector_path='D:\Programs\python3.8\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml'):
    face_cascade = cv2.CascadeClassifier(face_detector_path)
    for filename in os.listdir(directory_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(directory_path, filename)
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor, 5)
            cropped_faces = []
            for (x,y,w,h) in faces:
                roi_color = img[y:y+h, x:x+w]
                cropped_faces.append(roi_color)
            for i, cropped in enumerate(cropped_faces):
                cropped_filename = os.path.splitext(filename)[0] + f"_cropped{i}" + os.path.splitext(filename)[1]
                cropped_path = os.path.join(directory_path, cropped_filename)
                cv2.imwrite(cropped_path, cropped)

crop_all_faces('.\\trainingShort\\n000084')

