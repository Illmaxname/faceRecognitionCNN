import os
import cv2

def readAllImg(path, *suffix):
    try:
        s = os.listdir(path)
        resultArray = []
        img_name_list = []
        fileName = os.path.basename(path)
        resultArray.append(fileName)
        for i in s:
            if endwith(i, suffix):
                document = os.path.join(path, i)
                img = cv2.imread(document)
                img_name_list.append(i)
                resultArray.append(img)
    except IOError:
        print("Error")
    else:
        print("Done")
        return resultArray, img_name_list

def endwith(s, *endstring):
   resultArray = map(s.endswith, endstring)
   if True in resultArray:
       return True
   else:
       return False

if __name__ == '__main__':

  result = readAllImg(".\\photos", '.jpg')
  print(result[1])
  #cv2.namedWindow("Image")
  #cv2.imshow("Image", result[69])
  #cv2.waitKey(0)
  #cv2.destroyAllWindows()