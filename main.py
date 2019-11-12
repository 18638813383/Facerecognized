from tkinter import *
# 导入ttk
from PIL import Image ,ImageTk
from tkinter import ttk
from tkinter import filedialog
import facetrain
from queue import Queue
import cv2
from time import *
import os
import pygame
import threading
import pyttsx3
root = Tk()
root.title('人脸检测')
name,idcard,mobile,capture = StringVar(),StringVar(),StringVar(),StringVar()
collecttype = 2
capture.set('0')
face_detector = cv2.CascadeClassifier('D:\\Users\\41671\\AppData\\Local\\Continuum\\anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
imagefile = '' #原始图片全局变量
imgname = '' #打开图片的路径全局变量
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('face_trainer/trainer.yml')
#cascadePath = "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"
#faceCascade = cv2.CascadeClassifier(cascadePath)
#开始播放音乐
def startplaymusic():
    print(tasks)
    voiceplay('成功识别')
    media = '.\\media\\Alarm07.wav'
    pygame.mixer.init()
    pygame.mixer.music.load(media)
    pygame.mixer_music.play(1, 0)
    pygame.time.delay(5000)  # 等待1s让音乐播放完成
    # if tasks: tasks.pop()
    media_queue.get()
#语音播报
def voiceplay(voice):
    engine = pyttsx3.init()
    engine.say(voice)
    engine.runAndWait()
#启动音乐播放线程
tasks = []
media_queue = Queue(1)
def startthread():
    if media_queue.full():
        return
    print('-----------播放音乐线程开始启动-----------------')
    t = threading.Thread(target=startplaymusic, args=[])
    t.setDaemon(True)
    media_queue.put(t)
    t.start()
    print('thread %s ended. ' % t.name)
#打开图像
def open_img():
    global imgname
    imgname = filedialog.askopenfilename(title='打开单个文件',
                                     filetypes=[('png文件',"*.png"), ('jpg文件','*.jpg' ), ('所有文件','*')],  # 只处理的文件类型
                                     initialdir='.') # 初始目录
    print(imgname)
    tmp = cv2.imread(imgname)
    tmp = cv2.resize(tmp,(300,300))
    cv2.imwrite('tmp.png',tmp)
    global imagefile
    imagefile = PhotoImage(file='tmp.png')
    global collecttype
    collecttype =1
    ocv.create_image(150,150,image=imagefile)
#选择多个图片文件
def open_imgs():
    global imgname
    imgname = filedialog.askopenfilenames(title='选择多个文件',
                                     filetypes=[('jpg文件',"*.jpg"), ('png文件','*.png' ), ('所有文件','*')],  # 只处理的文件类型
                                     initialdir='.') # 初始目录
    global collecttype
    collecttype =3
#打开视频
def open_video():
    global videoname
    videoname = filedialog.askopenfilename(title='打开单个视频文件',
                                     filetypes=[('mp4文件',"*.mp4"), ('rmvb文件','*.rmvb' ), ('所有文件','*')],  # 只处理的文件类型
                                     initialdir='.') # 初始目录
    print(videoname)
    global collecttype
    collecttype =0
videoname = '' #打开视频文件名称
def opencapture():
    i = 0
    capture =cv2.VideoCapture(i)
    print('目前打开摄像头编号为',i,'按ESC键退出摄像头')
    print('您已选择摄像头采集样本，请确认')
    global collecttype
    collecttype = 2
# 人脸检测函数
def facedetect(image):
    img = cv2.imread(image)
    # 转为灰度图片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 检测人脸
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + w), (0, 255, 0))
    cv2.imshow('capture', img)
    k = cv2.waitKey(0)
#人脸数据收集,得到最大的userid值,根据身份证号判断是否为重复数据
def facedata(idcard):
    path = 'Facedata'
    faceid=0
    faceidlist= [] #所有的userid 列表
    idcardlist = [] #所有的idcard列表
    numlist = [] #序号列表
    duplicate = 0 #判断是否是重复的身份证号
    maxnum = 0
    try:
        for i in os.listdir(path):
            faceidlist.append(int(i.split(".")[2]))
            idcardlist.append(i.split(".")[3])
            if idcard == i.split(".")[3]:  #身份证号已存在
                faceid = int(i.split('.')[2])
                numlist.append(int(i.split(".")[4]))
                duplicate = 1
        if duplicate == 1:
            maxnum = max(numlist)
            print('********样本库中已有该样本，身份号为*************:',idcard)
            return faceid,maxnum
        else: #如果为新的身份证号，则判断为新人
            faceid=int((max(faceidlist)))+1
    except:
        faceid=0
    print('下一个将要采集的faceid是：',faceid)
    return faceid,maxnum
#摄像头采集,每次采集100个样本
def collectfromcapture(capturenum,name,idcard):
    print('-----------开始从摄像头读取图像------------------------')
    capturenum = int(capturenum)
    capture =cv2.VideoCapture(capturenum)
    print('\n 姓名:',name,'\n 身份证号:',idcard)
    path='Facedata'
    print('目前打开摄像头编号为',capturenum,'按ESC键退出摄像头')
    faceid,count = facedata(idcard)
    startnum = count
    if count == 0: #判断是否为新增的faceid
        count = int(faceid) * 1000  #每个人可以采集1000个样本，例如第一个人为1-1000，第2个人为1001-2000
        startnum = count
    print('10秒后，即将开始采集样本，请调整后姿势')
    sleep(10)
    while(True):
        ret,img=capture.read()
        cv2.putText(img,strftime("%y-%m-%d %H:%M:%S"),(10,70),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
        # 转为灰度图片
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 检测人脸
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + w), (0, 255, 0))
            count +=1
            cv2.imwrite("Facedata/capture." + str(name) + '.' + str(faceid) + '.' + str(idcard) + '.' + str(count) + '.jpg',gray[y: y + h, x: x + w])
        print('已经采集到的样本序号：',count)
        cv2.imshow('capture',img)
        k = cv2.waitKey(1)
        if k == 27:
            break
        elif int(count) >= int(startnum) + 100:  # 得到100个样本后退出摄像
            print('样本已经采集完成',name,count)
            break
    capture.release()
    cv2.destroyAllWindows()
#照片采集
def collectfromimg(img,name,idcard):
    print('-----------开始从照片中读取图像------------------------')
    faceid ,count = facedata(idcard) #得到将要采集的faceid,和最大的序号
    print('***********将要采集的faceid和序号是****************',faceid,count)
    if count == 0: #新增样本
        count = int(faceid) * 1000
    img = cv2.imread(img)
    # 转为灰度图片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 检测人脸
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    if len(faces) != 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + w), (0, 255, 0))
            count +=1
            cv2.imwrite("Facedata/image." + str(name) + '.' + str(faceid) + '.' + str(idcard) + '.' + str(count) + '.jpg',
                    gray[y: y + h, x: x + w])
            print('************样本已保存至Facedata目录****************')
            if int(count) >= int((int(faceid)+1)*1000):  # 得到100个样本后退出摄像
                print('样本已采集完成')
                break
    else:
        print('*****************未识别到人脸****************')
    print('----------------照片采集样本完成----------------------')
#多照片采集
def collectfromimgs(imgs,name,idcard):
    print('-----------开始从照片中读取图像------------------------')
    print('**********将要采集的照片列表为：',imgs)
    faceid ,count = facedata(idcard) #得到将要采集的faceid,和最大的序号
    print('***********将要采集的faceid和序号是****************',faceid,count)
    if count == 0: #新增样本
        count = int(faceid) * 1000
    for img in imgs:
        img = cv2.imread(img)
        # 转为灰度图片
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 检测人脸
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        if len(faces) != 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + w), (0, 255, 0))
                count +=1
                cv2.imwrite("Facedata/image." + str(name) + '.' + str(faceid) + '.' + str(idcard) + '.' + str(count) + '.jpg',
                        gray[y: y + h, x: x + w])
                print('************样本已保存至Facedata目录****************')
                if int(count) >= int((int(faceid)+1)*1000):  # 得到100个样本后退出摄像
                    print('样本已采集完成')
                    break
        else:
            print('*****************未识别到人脸****************')
        print('----------------照片采集样本完成----------------------')
#视频采集
def collectfromvideo(video,name,idcard):
    print('-----------开始从视频中读取图像,请确保该视频中只有一个人的图像------------------------')
    print('采集的视频名称，姓名，身份证号是：',video,name,idcard)
    path = 'Facedata'
    cam = cv2.VideoCapture(video)
    faceid,count = facedata(idcard) #根据身份证号判断样本库是否已存在
    startnum = count
    if count == 0: #判断是否为新增的faceid
        count = int(faceid) * 1000  #每个人可以采集1000个样本，例如第一个人为1-1000，第2个人为1001-2000
        startnum = count
    while(True):
        ret,img = cam.read()
        if ret:
            # 转为灰度图片
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 检测人脸
            faces = face_detector.detectMultiScale(gray, 1.2, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + w), (0, 255, 0)) #在图像上画矩形
                count += 1
                cv2.imwrite(
                    "Facedata/video." + str(name) + '.' + str(faceid) + '.' + str(idcard) + '.' + str(count) + '.jpg',
                    gray[y: y + h, x: x + w])
            print('已经采集到的样本序号：', count)
            cv2.imshow('capture', img)
            k = cv2.waitKey(1)
            if k == 27:
                break
            elif int(count) >= int(startnum) + 500:  # 得到500个样本后退出摄像
                print('样本已经采集完成', name, count)
                break
        else:
            print('样本文件打开失败')
            break
    cam.release()
#采集函数
def facecollect(type,name,idcard,mobile):
    print('----------采集类型-姓名-身份证号-手机号：',type,name,idcard,mobile)
    if len(name) == 0  or len(idcard) == 0 or len(mobile) == 0:
        print('请检查姓名，身份证号，手机号是否已经设置')
        return
    else:
        if type == 2:
            print('采集类型为摄像头')
            collectfromcapture(capture.get(),name,idcard)
        elif type == 1:
            print('采集类型为图片')
            collectfromimg(imgname,name,idcard)
        elif type == 0:
            print('采集类型为视频')
            collectfromvideo(videoname,name,idcard)
        elif type == 3:
            print('采集类型为多图片采集')
            collectfromimgs(imgname, name, idcard)
        else:
            print('采集失败，重新选择')
            return
    #facedetect(result)
names = []
#初始化姓名列表,按照faceid生成姓名列表
def initnames():
    path = 'Facedata'
    n = 0 #inital names
    limitn = 1000 #inital name num
    tmplist = []
    global names
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]  # join函数的作用？
    for i in imagePaths:
        tmp = os.path.split(i)[-1].split(".")[1:3]
        tmplist.append(tmp)
    for new in tmplist:
        #print(new)
        if new not in names:
            names.append(new)
    #print('-------:', names)
    names.sort(key=lambda x: x[1])
#人脸图像识别
modelresult =''  #比对结果全局变量
def face_recognitionfromimg(imgfile):
    print('------临时测试---------')
    imgfile = cv2.imread(imgfile)
    gray = cv2.cvtColor(imgfile, cv2.COLOR_BGR2GRAY) #二值化
    minW = 0.1 * imgfile.shape[1]
    minH = 0.1 * imgfile.shape[0]
    initnames()
    #cv2.imshow('aa',gray)
    #cv2.waitKey(0)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,#每次图像尺寸减小的比例
        minNeighbors=1, #每个目标至少要检测到的次数,图像的时候设置为1
        minSize=(int(minW), int(minH))#目标的最小尺寸
    )
    print('识别到的人脸数量是：',len(faces))
    #if len(faces) == 1:
    for (x, y, w, h) in faces:
        cv2.rectangle(imgfile, (x, y), (x + w, y + h), (0, 255, 0), 2)
        idnumtmp, confidence = recognizer.predict(gray[y:y + h, x:x + w])  # 预测
        print('INFO:idnum:', idnumtmp, 'confidence:', confidence)
        file = open('result.txt', mode='a', encoding='UTF-8')
        if len(faces) == 1: #识别到照片上只有一张人脸的时候
            if confidence < 80:
                #print(int(idnum / 100))
                idnum = names[int(idnumtmp / 1000)]
                print(idnum,idnumtmp)
                modelname =''
                confidence = "{0}%".format(round(100 - confidence))
                for i in os.listdir('.\\Facedata'): #查找样本的文件名
                    m = i.split('.')[4]
                    if m == str(idnumtmp) :
                        print(m)
                        modelname = i
                        break
                tmp = cv2.imread('Facedata\\'+modelname)
                tmp = cv2.resize(tmp,(300,300))
                cv2.imwrite('modeltmp.png',tmp)
                global modelresult
                modelresult = PhotoImage(file='modeltmp.png')
                modelocv.create_image(150,150,image=modelresult)
                global name1,idcadr1,consider
                name1.set(modelname.split('.')[1]),idcard1.set(modelname.split('.')[3]),consider.set(confidence)
            else:
                idnum = "unknown"
                confidence = "{0}%".format(round(100 - confidence))
                print(idnum)
        else:
            if confidence > 0:
                print('该图片识别到多个人脸，检测结果，见文件result.txt')
                confidence = "{0}%".format(round(100 - confidence))
                modelname =''
        #        confidence = "{0}%".format(round(100 - confidence))
                for i in os.listdir('.\\Facedata'): #查找样本的文件名
                    m = i.split('.')[4]
                    if m == str(idnumtmp) :
                        print(m)
                        modelname = i
                        break
                file.write(imgname + '\t' + str(idnumtmp) + '\t' + modelname + '\t' + confidence + '\n')
        file.close()
#人脸摄像头识别
def face_recognitionfromcapture(capturenum):
    print('---------------摄像头人脸识别开始--------------')
    #initnames() #根据样本文件形成，样本序号列表
    cam = cv2.VideoCapture(int(capturenum))
    minW = 0.1*cam.get(3) #宽度
    minH = 0.1*cam.get(4) #高度
    font = cv2.FONT_HERSHEY_SIMPLEX
    num =1  #结果文件初始序列号
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    outputfilename = './resultfile/capture_result_' + strftime("%Y%m%d%H%M") + '.avi'
    out = cv2.VideoWriter(outputfilename, fourcc, 25.0, (int(cam.get(3)), int(cam.get(4))))
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #二值化
        #cv2.resizeWindow(img,800,600)
        cv2.putText(img, strftime("%Y-%m-%d %H:%M:%S"), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA) #图像打水印
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH))
        )
        filename = './resultfile/capture_result_' + strftime("%Y%m%d") + '_' + str(num) + '.txt'
        try:
            count = len(open(filename, 'r').readlines())
            if count >= 10000:  #每个文件最多保存10000行数据
                num += 1
                filename = './resultfile/capture_result_' + strftime("%Y%m%d") + '_' + str(num) + '.txt'
        except:
            print('***********文件不存在，或打开异常*************')
        file = open(filename, mode='a', encoding='UTF-8')
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            idnumtmp, confidence = recognizer.predict(gray[y:y+h, x:x+w]) #预测
            #print('INFO:idnum:',idnumtmp,'confidence:',confidence)
            if confidence < 100:
                confidence = "{0}%".format(round(100 - confidence))
                modelname = ''
                for i in os.listdir('.\\Facedata'): #查找样本的文件名
                    m = i.split('.')[4]
                    if m == str(idnumtmp) :
                        #print(m)
                        modelname = i
                        idnum = modelname.split('.')[1]
                        tmp = cv2.imread('Facedata\\'+modelname)
                        tmp = cv2.resize(tmp,(100,100))
                        img[0:100,img.shape[1]-100:img.shape[1]]=tmp #右上角显示命中的样本
                        cv2.putText(img, modelname.split('.')[1] + '-' + modelname.split('.')[3] +  '-' + confidence , (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                            cv2.LINE_AA)
                        break
                if int(confidence[:-1])>50:
                    #print(confidence[:-1])
                    #voiceplay('成功识别')
                    print('相似度达到50%以上，开始播放音乐！！！！，现在的相似度是',idnumtmp,modelname.split('.')[1],confidence)
                    startthread()
                    print('-----------------------------------------')
                file.write(strftime("%Y-%m-%d %H:%M:%S") + '\t' + modelname + '\t' + confidence + '\n')
            else:
                idnum = "未识别"
                confidence = "{0}%".format(round(100 - confidence))

            cv2.putText(img, str(idnum), (x+5, y-5), font, 1, (0, 0, 255), 1)
            cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (0, 255, 0), 1)

        cv2.imshow('camera', img)
        out.write(img)
        k = cv2.waitKey(10)
        if k == 27:
            break
    file.close()
    cam.release()
    out.release()
    cv2.destroyAllWindows()
    print('***************************************摄像头人脸识别已退出*****************************')
#人脸视频识别
def face_recognitionfromvideo(videofile):
    print('---------------开始从视频中识别人脸------------------')
    cam = cv2.VideoCapture(videofile)
    initnames()
    minW = 0.1*cam.get(3) #宽度
    minH = 0.1*cam.get(4) #高度
    font = cv2.FONT_HERSHEY_SIMPLEX
    while(True):
        ret,img = cam.read()
        if(ret):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 二值化
                # cv2.resizeWindow(img,800,600)
            cv2.putText(img, strftime("%Y-%m-%d %H:%M:%S"), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                            cv2.LINE_AA)  # 图像打水印
            faces = face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=5,
                    minSize=(int(minW), int(minH))
                )
            file = open('video_result.txt', mode='a', encoding='UTF-8')
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                idnumtmp, confidence = recognizer.predict(gray[y:y + h, x:x + w])  # 预测
                    # print('INFO:idnum:',idnumtmp,'confidence:',confidence)

                if confidence < 100:
                        # print(int(idnum/100))
                        # idnum = names[int(idnumtmp/1000)]
                        # print(idnum)
                    confidence = "{0}%".format(round(100 - confidence))
                    for i in os.listdir('.\\Facedata'):  # 查找样本的文件名
                        m = i.split('.')[4]
                        if m == str(idnumtmp):
                                # print(m)
                            modelname = i
                            break
                    idnum = modelname.split('.')[1]
                    tmp = cv2.imread('Facedata\\' + modelname)
                    tmp = cv2.resize(tmp, (100, 100))
                        # print(cam.get[4])
                    img[0:100, img.shape[1] - 100:img.shape[1]] = tmp  # 右上角显示命中的样本
                    cv2.putText(img, modelname.split('.')[1] + '-' + modelname.split('.')[3] + '-' + confidence,
                                    (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                                    cv2.LINE_AA)
                    if int(confidence[:-1]) > 50:
                        # print(confidence[:-1])
                        print('相似度达到50%以上，开始播放音乐！！！！，现在的相似度是', confidence)
                        pygame.mixer_music.play(1, 0)
                        pygame.time.delay(1000)  # 等待1s让音乐播放完成
                        file.write(strftime("%Y-%m-%d %H:%M:%S") + '\t' + modelname + '\t' + confidence + '\n')
                else:
                    idnum = "未识别"
                    confidence = "{0}%".format(round(100 - confidence))
                    cv2.putText(img, str(idnum), (x + 5, y - 5), font, 1, (0, 0, 255), 1)
                    cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (0, 255, 0), 1)
            cv2.imshow('video', img)
            k = cv2.waitKey(10)
            if k == 27: #Esc键退出当前窗口
                break
        else:
            print('--------------------视频处理完成退出视频,点击Esc键退出视频播放-------------------')
            k = cv2.waitKey(1000)
            if k == 27: #Esc键退出当前窗口
                break
            cam.release()
    file.close()
    cam.release()
    cv2.destroyAllWindows()


#人脸识别
def face_recognition(recognitiontypye):
    print('---------------人脸识别比对开始-----------------------')
    print('----------------开始参数检测-------------------------') #默认选择摄像头比对
    if recognitiontypye == 0 :
        print('比对类型为视频比对，请确认')
        face_recognitionfromvideo(videoname)
    elif recognitiontypye == 1:
        print('比对类型为图像比对，请确认')
        print('需要比对图片名称：',imgname)
        face_recognitionfromimg(imgname)
        print('----------------')
    elif recognitiontypye == 2:
        print('比对类型摄像头比对,请确认')
        print(capture.get())
        face_recognitionfromcapture(capture.get())
    else :
        print('参数错误，请检测,采集类型为：',recognitiontypye)
#root.geometry('1000x600')
#创建frame
frame1 = ttk.Frame(root)
frame1.pack(side=TOP)
#创建laberframe
lf_frame1 = ttk.Labelframe(frame1,text ='样本采集')
lf_frame1.pack(side=LEFT)
lf1_frame1 = ttk.Labelframe(lf_frame1,text='参数输入')
lf1_frame1.pack(side=TOP,padx=10,pady=10)
ttk.Label(lf1_frame1,text='姓名：').pack(side=LEFT,padx=10,pady=10)
ttk.Entry(lf1_frame1,textvariable=name).pack(side=LEFT,padx=10,pady=10)
ttk.Label(lf1_frame1,text='身份证号：').pack(side=LEFT,padx=10,pady=10)
ttk.Entry(lf1_frame1,textvariable=idcard).pack(side=LEFT,padx=10,pady=10)
ttk.Label(lf1_frame1,text='手机号：').pack(side=LEFT,padx=10,pady=10)
ttk.Entry(lf1_frame1,textvariable=mobile).pack(side=LEFT,padx=10,pady=10)

lf2_frame1 = ttk.Labelframe(lf_frame1,text='摄像头选择')
lf2_frame1.pack(side=TOP,padx=10,pady=10)
ttk.Radiobutton(lf2_frame1,text='摄像头 0',value=0,variable=capture).pack(side=LEFT,padx=50,pady=10)
ttk.Radiobutton(lf2_frame1,text='摄像头 1',value=1,variable=capture).pack(side=LEFT,padx=50,pady=10)
ttk.Radiobutton(lf2_frame1,text='摄像头 2',value=2,variable=capture).pack(side=LEFT,padx=50,pady=10)
ttk.Radiobutton(lf2_frame1,text='摄像头 3',value=3,variable=capture).pack(side=LEFT,padx=50,pady=10)

imglabelframe = ttk.Labelframe(lf_frame1,text='图片比对')
imglabelframe.pack()
ttk.Label(imglabelframe,text='原始图片:').pack(side=LEFT,padx=10,pady=10)
ocv = Canvas(imglabelframe,background='white',width=300,height=300)
#ocv.create_image(10, 10, image=imagefile)
ocv.pack(side=LEFT,padx=10,pady=10)
#orignalimg =
ttk.Label(imglabelframe,text='命中样本:').pack(side=LEFT,padx=10,pady=10)
modelocv = Canvas(imglabelframe,background='white',width=300,height=300)
modelocv.pack(side=LEFT,padx=10,pady=10)

resultlabelframe = ttk.Labelframe(lf_frame1,text='命中结果')
resultlabelframe.pack()
result = ['张三','410113123','75%']
name1,idcard1,consider = StringVar(),StringVar(),StringVar()
name1.set(result[0]),idcard1.set(result[1]),consider.set(result[2])
ttk.Label(resultlabelframe,text='姓名：').pack(side=LEFT,padx=10,pady=10)
ttk.Entry(resultlabelframe,textvariable=name1).pack(side=LEFT,padx=40,pady=10)
ttk.Label(resultlabelframe,text='身份证号：').pack(side=LEFT,padx=10,pady=10)
ttk.Entry(resultlabelframe,textvariable=idcard1).pack(side=LEFT,padx=40,pady=10)
ttk.Label(resultlabelframe,text='相似度：').pack(side=LEFT,padx=10,pady=10)
ttk.Entry(resultlabelframe,textvariable=consider).pack(side=LEFT,padx=40,pady=10)


oplabelframe = ttk.Labelframe(frame1,text='功能区')
oplabelframe.pack(side=LEFT,padx=20,pady=10)
lf1_frame2 = ttk.Labelframe(oplabelframe,text='样本采集')
lf1_frame2.pack(side=TOP)
ttk.Button(lf1_frame2,text='摄像头',command=opencapture).pack(side=TOP,padx=10,pady=10)
ttk.Button(lf1_frame2,text='单图片',command=open_img).pack(side=TOP,padx=10,pady=10)
ttk.Button(lf1_frame2,text='多图片',command=open_imgs).pack(side=TOP,padx=10,pady=10)
ttk.Button(lf1_frame2,text='视频',command=open_video).pack(side=TOP,padx=10,pady=10)
ttk.Button(lf1_frame2,text='采集',command= lambda :facecollect(collecttype,name.get(),idcard.get(),mobile.get())).pack(side=TOP,padx=10,pady=10)

train_labelframe = ttk.Labelframe(oplabelframe,text='样本训练')
train_labelframe.pack(side=TOP)
ttk.Button(train_labelframe,text='训练样本',command=facetrain.facetrain).pack(side=TOP,padx=10,pady=50)
#ttk.Button(train_labelframe,text='图片').pack(side=TOP,padx=10,pady=10)
#ttk.Button(train_labelframe,text='视频').pack(side=TOP,padx=10,pady=10)

cmpare_laberframe = ttk.Labelframe(oplabelframe,text='识别比对')
cmpare_laberframe.pack(side=TOP)
ttk.Button(cmpare_laberframe,text='摄像头',command=opencapture).pack(side=TOP,padx=10,pady=10)
ttk.Button(cmpare_laberframe,text='图片',command=open_img).pack(side=TOP,padx=10,pady=10)
ttk.Button(cmpare_laberframe,text='视频',command=open_video).pack(side=TOP,padx=10,pady=10)
ttk.Button(cmpare_laberframe,text='比对',command= lambda : face_recognition(collecttype)).pack(side=TOP,padx=10,pady=10)

root.mainloop()

#root.mainloop()