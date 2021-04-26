import cv2 as cv
import numpy as np

#Pretrained data
face_detector = cv.CascadeClassifier('face_detection.xml')
smile_detector=cv.CascadeClassifier('smile_detection.xml')
webcam=cv.VideoCapture(0)

#Running in realtime
while True:
    #Reading frames from camera and transforming them in black&white
    successful_frame,frame = webcam.read()
    blank = np.zeros(frame.shape[:2],dtype='uint8')
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    
    #Detecting faces on gray img
    faces = face_detector.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=6)
    
    #Drawing rect on face
    for (x,y,w,h) in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness=1)
        
        #Cutting img just on frame(actually making everything other white)
        mask = cv.rectangle(blank,(x+(w//3),y+(h//3)),(x+w,y+h),(255,255,255),thickness=-1)
        face = cv.bitwise_and(gray,gray,mask=mask)

        #Detecting smiles on face
        smile = smile_detector.detectMultiScale(face,1.2,10)

        #Debuging smiles detection by drawing rects around smile
        #for (x,y,w,h) in smile:
            #cv.rectangle(face,(x,y),(x+w,y+h),(0,0,255),thickness=2)
        
        if len(smile)>0:
            cv.putText(frame,"HA HA You are smiling",(x,y+h+50),cv.FONT_ITALIC,0.5,(230,50,30),2,)
        
        if len(smile)==0:
            cv.putText(frame,"WHY SO SERIOUS",(x,y+h+20),cv.FONT_ITALIC,0.5,(0,0,255),2,)


    #Displaying
    cv.imshow('Face Detector',frame)
    
    #Stop running if SPACABAR is pressed
    key = cv.waitKey(1)
    if key==32:
        break


#Close all windows and turn off webcam when it's done
webcam.release()
cv.destroyAllWindows()