import cv2
import handTrackingModule as htm
import mediapipe
import os
import time

wCam , hCam = 640, 480

prevtime =0

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

folderPath = "fingerImages"
myList = os.listdir(folderPath)
overLayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overLayList.append(image)

detector = htm.handDectector(detectionCon=0.8)

tipIDS = [ 4 , 8, 12, 16, 20]

while True:
    success , img = cap.read()
    
    img = detector.findHands(img)
    lmList = detector.findPosition(img ,draw = False)
    # print(lmList)

    if len(lmList)!=0:
        fingers = []
        #for right thumb
        if lmList[tipIDS[0]][1]> lmList[tipIDS[0]-1][1]: 
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1,5):  
            if lmList[tipIDS[id]][2]< lmList[tipIDS[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        #print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)

        h,w,c = overLayList[totalFingers-1].shape
        img[0:h ,0:w] = overLayList[totalFingers-1]

        cv2.rectangle(img , (20 , 225) , (170,425) , (0,255,0) ,cv2.FILLED)
        cv2.putText(img , str(totalFingers) ,(45,375) , cv2.FONT_HERSHEY_PLAIN , 10 , (255, 0 ,255) ,15 )

    currtime = time.time()
    fps = 1/(currtime -prevtime)
    prevtime = currtime

    cv2.putText(img , f'FPS:{int(fps)}' , (400,70) , cv2.FONT_HERSHEY_PLAIN , 2, (255,0,0),2 )

    cv2.imshow("Images",img)

    if cv2.waitKey(10) == ord('q'):
        break