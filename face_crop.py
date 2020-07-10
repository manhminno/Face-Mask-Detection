import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades  + 'haarcascade_frontalface_default.xml')

def face_crop(path_link):
    img = cv2.imread(path_link)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x, y-60), (x+w, y+h), (255,0,0), 2)
        img2 = img[y-60+2:y+h-2, x+2:x+w-2]
    
    
