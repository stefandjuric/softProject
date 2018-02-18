import cv2
import numpy as np
from keras.models import load_model

model = load_model('model.h5')

def houdhTransform(frame):
    #---------------------------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # cv2.imshow("hsv",hsv)
    # cv2.waitKey(0)

    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    #cv2.imshow('res',res)
    #cv2.waitKey(0)
    gray1 = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)


    jjj = cv2.erode(gray1, None, iterations=1)
    #cv2.imshow('res',jjj)
    #cv2.waitKey(0)


    minLineLength = 600
    maxLineGap = 8
    lines = cv2.HoughLinesP (jjj, 1, np.pi / 180, 40, minLineLength, maxLineGap)
    #lines = cv2.HoughLinesP (edges, 1, np.pi / 180, 40, minLineLength, maxLineGap) #++++++++++

    minx = lines[0][0][0]   #postavljanje inicijalnih vrijednosti moze i rucno
    miny = lines[0][0][1]
    maxx = lines[0][0][2]
    maxy = lines[0][0][3]

    temp = len(lines)

        #cv2.imshow("test",frame)
        #cv2.waitKey()
    #print "start"

    for i in  range(temp):  #trazimo najduzu liniju koja treba da bude linija sa videa kroz koju prolaze brojevi
        x1 = lines[i][0][0]
        y1 = lines[i][0][1]
        x2 = lines[i][0][2]
        y2 = lines[i][0][3]
        if  x1 < minx:
            miny = y1
            minx = x1
        if  x2 > maxx:
            maxx = x2
            maxy = y2
    #cv2.line(frame, (minx,miny), (maxx, maxy), (0, 255, 0), 2)
    #cv2.startWindowThread()
    #cv2.namedWindow("preview")
    #cv2.imshow('preview',frame)
    #cv2.waitKey(0)
    return minx, miny, maxx, maxy

def openVideo(videoName):
    vid = cv2.VideoCapture(videoName)
    if(vid.isOpened()):
        ret, frame = vid.read()
        return ret, frame
    else:
        print cv2.__version__
        return 0


lastX=None
lastY=None
frameNumber=None
currentFrame=0
list = []

def locatedOnVector(vector,dot1,dot2):
    global lastX
    global lastY
    global frameNumber
    global currentFrame
    global list

    k = float(vector[1]-1-vector[3]-1)/float(vector[0]-1-vector[2]-1)
    n = float(vector[1]-1 - k*(vector[0]-1))

    stop = isSameNumber(dot2)
    if(stop==False):
        return False

    if(vector[0]<vector[2]):
        if((vector[0]<dot1[0] and vector[2]>dot1[0]) or (vector[0]<dot2[0] and vector[2]>dot2[0])):
            if(abs(dot2[1] - abs(int((k*dot2[0]) + n)))<2):
                lastX = dot2[0]
                lastY = dot2[1]
                frameNumber=currentFrame
                list.insert(0,(dot2[0],dot2[1],currentFrame))
                return True
            else:
                return False

def isSameNumber(dot2):
    j=0
    for i in list:
        frameNumber = i[2]
        lastX = i[0]
        lastY = i[1]
        if(frameNumber!=None):
            if(abs(currentFrame-frameNumber)<=20):
                if(abs(lastX-dot2[0])<=15 and abs(lastY-dot2[1])<=15):
                    return False
        j += 1
        if(j==10): break


def getNumbers(contures):
    numbers = []
    for conture in contures:
        num = cv2.boundingRect(conture)
        if(num[3]>10):
            numbers.append(cv2.boundingRect(conture))
    return numbers



def creatingImageForPredicting(image):
    imageForPreparing = np.zeros((28, 28));
    height,width = image.shape
    currentSize = (28-height)/2
    for i in range(0, height):
        for j in range(0, width):
            if(currentSize+i>=28 or currentSize+j>=28):
                currentSize=currentSize-1
            imageForPreparing[i+currentSize, j+currentSize] = image[i, j]
    return imageForPreparing


#izdvajanje samo brojeva(crno-bijela slika sa brojevima)
def preparingImageForFindingNumber(image):
    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sensitivity = 75
    lower_white = np.array([0,0,255-sensitivity])
    upper_white = np.array([255,sensitivity,255])

    mask = cv2.inRange(hsvImage, lower_white, upper_white)
    result = cv2.bitwise_and(image,image, mask= mask)

    grayImage= cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    grayImage= cv2.GaussianBlur(grayImage, (5, 5), 0)

    cannyImage = cv2.Canny(grayImage, 50, 100)
    cannyImage = cv2.dilate(cannyImage, None, iterations=1)
    cannyImage = cv2.erode(cannyImage, None, iterations=1)
    return cannyImage



def runVideo():
    global currentFrame
    global frameNumber
    global list
    sumList=[]
    video = "./data/video-"
    for x in range(0, 10):
        sum = 0
        videoName = "{0}{1}{2}".format(video, str(x), ".avi")
        vid = cv2.VideoCapture(videoName)
        frameNumber = None
        currentFrame = 0
        list = []
        ret, frame = vid.read()
        while(vid.isOpened()):
            currentFrame += 1
            x1, y1 , x2, y2 = houdhTransform(frame)

            #izdvajanje samo brojeva(crno-bijela slika sa brojevima)
            #---------------------------------
            cannyImage = preparingImageForFindingNumber(frame)
            #dobijamo slike pojedinacnik brojeva tj izdvajamo odblasti
            im2, cnts, hierarchy = cv2.findContours(cannyImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            numbers = getNumbers(cnts)

            for number in numbers:
                leng = int(number[3])
                pt1 = int(number[1] + number[3] // 2 - leng // 2)
                pt2 = int(number[0] + number[2] // 2 - leng // 2)
                roi = cannyImage[pt1:pt1+leng, pt2:pt2+leng]

                dot1 = [number[0],number[1]]
                dot2 = [number[0]+number[2],number[1]+number[3]]
                onVector = locatedOnVector((x1,y1,x2,y2),dot1,dot2)

                #preparing image for predict
                imageForPreparing = creatingImageForPredicting(roi)
                # cv2.imshow("dsada",imageForPreparing)
                # cv2.waitKey(0)
                prediction = model.predict(imageForPreparing.reshape(1, 784), verbose=1)
                value = np.argmax(prediction)
                if(onVector and x2>dot2[0]):
                    sum = sum + value
                    #cv2.line(frame, (x1,y1), (x2, y2), (0, 255, 0), 2)
                    #cv2.imshow("jjj",frame)
                    #cv2.waitKey(0)
                    #cv2.imshow("aaaa",roi)
                    #cv2.waitKey(0)
                    #print "sumaaaa    ",sum
                    #cv2.destroyAllWindows()
            ret, frame = vid.read()
            if(ret!=True):
                sumList.append((videoName,sum))
                break
            #print "reeeeeeet", ret
    createOutputFile(sumList)



def createOutputFile(sumList):
    with open('./data/out.txt', 'w') as the_file:
        the_file.write('SW 19/2014 Stefan Djuric\n')
        the_file.write('file	sum\n')
        for sum in sumList:
            s = sum[0].split("/")
            print s
            the_file.write("{0}{1}{2}{3}".format(s[2], "\t", sum[1],"\n"))
        the_file.close()

runVideo()
