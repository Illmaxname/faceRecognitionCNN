# faceRecognitionCNN
Face recognition system based on CNN model. Recognition accuracy on the test training dataset was achieved in 76%
## Libs
The libraries OpenCV, sklearn and TensorFlow were used for development.
## Dataset
The training sample included 4-5 individuals, 50 individuals from VGG2Face dataset and as a result of photo augmentation, the final dataset was approximately 1000 images.
The DataSet class from **dataSet.py** is a dataset that is used to train convolutional neural network models (CNN). this class provides the functionality to retrieve, augmentation and formatting of image data, as well as storing this data in a convenient format. Formatting data in this case is a safety net against specifying data of the wrong format, such as size or color channel.
## Preprocessing
**pick_face.py** processes images: compresses to 128 by 128, grays out and crops the face using haar cascades. The **crop_face.py** is used for manual processing if the Haar cascades did not reveal the face in the photo.
## Train model
The Model class from **train_model.py** represents a face recognition model and contains methods for building a neural network, training, testing, and saving.
The image shows the process of training the model
![image](https://github.com/Illmaxname/faceRecognitionCNN/assets/81902786/042e6789-6869-42fc-8561-690087efcd86)
## Test model
Two programs were used to test the trained CNN model in the photographs: **test_model.py** and **read_camera.py**. In the read_camera.py, if the probability is less than 75%, then the face is marked as "Unknown". Also, a frame is drawn along the borders of the face and the most probable class or “Unknown” is signed.
