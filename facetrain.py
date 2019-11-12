import numpy as np
from PIL import Image
import os
import cv2
# 人脸数据路径
path = 'Facedata'
names= []
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('D:\\Users\\41671\\AppData\\Local\\Continuum\\anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')

#detector = cv2.CascadeClassifier("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]  # join函数的作用？
    faceSamples = []
    ids = []
    #print(imagePaths)
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')   # convert it to grayscale
        img_numpy = np.array(PIL_img, 'uint8')
        tmp = os.path.split(imagePath)[-1].split(".")
        id = int(os.path.split(imagePath)[-1].split(".")[4])
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x: x + w])
            ids.append(id)
    return faceSamples, ids

def facetrain():
    print('---------------开始执行样本训练程序，等待完成后即可进行识别比对----------')
    print('Training faces. It will take a few seconds. Wait ...')
    faces, ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))
    print('ids is:',np.array(ids))
    recognizer.write(r'face_trainer/trainer.yml')
    print("{0} faces trained. Exiting Program".format(len(np.unique(ids))))
    print('---------------样本训练完成，开始识别比对操作-------------------------')
if __name__ == '__main__':
    print('start facetrain')
    #facetrain()
    faces ,ids = getImagesAndLabels(path)
    print(np.array(ids))
    print("{0} faces trained. Exiting Program".format(len(np.unique(ids))))
else:
    print('样本训练模块已经导入')