import cv2
import numpy as np
from keras.models import load_model

model = load_model('model.h5')

def houdhTransform(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2,2),np.uint8) #kernel prelazi preko slike moze i (5x5)


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

def getNumber(array):
    for i in range(0, 9):
        print array[0][i]
        if(array[0][i]==1):
            return i


def getNormalLine(x,y,vector):
    k1 = (vector[1]-vector[3])/(vector[0]-vector[2])
    k2 = float(-1/k1)
    n2 = y-k2*x
    return k2,n2

def intersection(k,n,vector):
    k1 = (vector[1]-vector[3])/(vector[0]-vector[2])
    n1 = vector[1] - k1*vector[0]
    x = (n1 - n)/(k-k1)
    y = k1*x + n1
    return x, y




lastX=None
lastY=None
frameNumber=None
currentFrame=0

import collections
list = []

def locatedOnVector(vector,dot1,dot2):
    global lastX
    global lastY
    global frameNumber
    global currentFrame
    global list

    #print "lastX     ",lastX
    #print "lastY      ",lastY
    k1,n1 = getNormalLine(dot1[0],dot1[1],vector)
    k2,n2 = getNormalLine(dot2[0],dot2[1],vector)

    x1,y1 = intersection(k1,n1,vector)
    x2,y2 = intersection(k2,n2,vector)


    k = float(vector[1]-1-vector[3]-1)/float(vector[0]-1-vector[2]-1)
    n = float(vector[1]-1 - k*(vector[0]-1))

    x0 = (dot1[0]+dot2[0])/2
    y0 = (dot1[1]+dot2[1])/2

    # print 'dot1     ',dot1
    # print 'dot2     ',dot2
    # print vector
    # print "x0----------- ",x0
    # print "y0----------- ",y0

    # print "k     ",k
    # print "n     ",n

    j=0
    for i in list:
        #print i
        frameNumber = i[2]
        lastX = i[0]
        lastY = i[1]
        if(frameNumber!=None):
            if(abs(currentFrame-frameNumber)<=20):
                if(abs(lastX-dot2[0])<=15 and abs(lastY-dot2[1])<=15):
                    return False
        j += 1
        if(j==10): break

    if(vector[0]<vector[2]):
        if((vector[0]<dot1[0] and vector[2]>dot1[0]) or (vector[0]<dot2[0] and vector[2]>dot2[0])):
            if(abs(dot2[1] - abs(int((k*dot2[0]) + n)))<2):
                # cv2.line(img,(vector[0],vector[1]),(vector[2],vector[3]),(255,0,0),5)
                # cv2.line(img,(dot2[0],dot2[1]),(dot1[0],dot1[1]),(255,0,0),5)
                # cv2.imshow("adasd",img)
                # cv2.waitKey(0)
                lastX = dot2[0]
                lastY = dot2[1]
                frameNumber=currentFrame
                list.insert(0,(dot2[0],dot2[1],currentFrame))
                return True
            else:
                return False
    else:
        if((vector[2]<x1 and vector[0]>x1) or (vector[2]<x2 and vector[0]>x2)):
            if(abs(dot2[1] - abs(int((k*dot2[0]) + n)))<2):
                lastX = dot2[0]
                lastY = dot2[1]
                frameNumber=currentFrame
                list.insert(0,(dot2[0],dot2[1],currentFrame))
                return True
            else:
                return False



def prikazVidea():
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
        #print "--------------------------", ret
        while(vid.isOpened()):
            currentFrame += 1
            # print "shape    ",frame.shape

            #prikaz svakog frama
            # cv2.imshow("jjj",frame)
            # cv2.waitKey(0)

            x1, y1 , x2, y2 = houdhTransform(frame)

            #izdvajanje samo brojeva(crno-bijela slika sa brojevima)
            #---------------------------------
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # cv2.imshow("hsv",hsv)
            # cv2.waitKey(0)

            sensitivity = 75
            lower_white = np.array([0,0,255-sensitivity])
            upper_white = np.array([255,sensitivity,255])

            mask = cv2.inRange(hsv, lower_white, upper_white)
            res = cv2.bitwise_and(frame,frame, mask= mask)
            # cv2.imshow('res',res)
            # cv2.waitKey(0)

            mojaSlika = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

            mojaSlika = cv2.GaussianBlur(mojaSlika, (5, 5), 0)

            image_edged = cv2.Canny(mojaSlika, 50, 100)
            image_edged = cv2.dilate(image_edged, None, iterations=1)
            image_edged = cv2.erode(image_edged, None, iterations=1)
            # cv2.imshow('mojaSlika2',image_edged)
            # cv2.waitKey(0)

            #----------------------------------


            #dobijamo slike pojedinacnik brojeva tj izdvajamo odblasti
            im2, cnts, hierarchy = cv2.findContours(image_edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            rects = [cv2.boundingRect(ctr) for ctr in cnts]

            for rect in rects:
                if(rect[3]>10):
                    leng = int(rect[3] * 1.6)
                    leng = int(rect[3])
                    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
                    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
                    roi = image_edged[pt1:pt1+leng, pt2:pt2+leng]

                    dot1 = [rect[0],rect[1]]
                    dot2 = [rect[0]+rect[2],rect[1]+rect[3]]
                    onVector = locatedOnVector((x1,y1,x2,y2),dot1,dot2)
                    # print "da li je na vektoru ",onVector
                    #print x1,y1,x2,y2
                    #print dot1
                    #print dot2

                    #preparing image for predict
                    imageForPreparing = np.zeros((28, 28));
                    h,w = roi.shape
                    o = (28-h)/2
                    for i in range(0, h):
                        for j in range(0, w):
                            if(o+i>=28 or o+j>=28):
                                o=o-1
                            imageForPreparing[i+o, j+o] = roi[i, j]
                    # cv2.imshow("dsada",imageForPreparing)
                    # cv2.waitKey(0)

                    #imageForPredict = roi.range(10, dtype=float).resize(28,28)
                    prediction = model.predict(imageForPreparing.reshape(1, 784), verbose=1)
                    # print prediction
                    #print getNumber(prediction)
                    value = np.argmax(prediction)
                    # print value
                    if(onVector and x2>dot2[0]):
                        sum = sum + value
                        #cv2.line(frame, (x1,y1), (x2, y2), (0, 255, 0), 2)
                        #cv2.imshow("jjj",frame)
                        #cv2.waitKey(0)
                        #cv2.imshow("aaaa",roi)
                        #cv2.waitKey(0)

                    #print "sumaaaa    ",sum
                    cv2.destroyAllWindows()
            ret, frame = vid.read()
            if(ret!=True):
                sumList.append((videoName,sum))
                break
            #print "reeeeeeet", ret
    createOutputFile(sumList)


def createOutputFile(sumList):
    with open('./data/out.txt', 'w') as the_file:
        the_file.write('RA X/2013 Marko Markovic\n')
        the_file.write('file	sum\n')
        for sum in sumList:
            s = sum[0].split("/")
            print s
            the_file.write("{0}{1}{2}{3}".format(s[2], "\t", sum[1],"\n"))
        the_file.close()

prikazVidea()
