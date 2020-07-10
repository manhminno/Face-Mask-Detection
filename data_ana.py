import os
import numpy as np
import cv2
# count = 0
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades  + 'haarcascade_frontalface_default.xml')

def face_crop(path_link, link, name):
    print(path_link)
    img = cv2.imread(path_link)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(faces)
    for (x,y,w,h) in faces:
        # cv2.rectangle(img, (x, y-60), (x+w, y+h), (255,0,0), 2)
        img = img[y:y+h, x:x+w]
        os.remove(path_link)
        cv2.imwrite(name, img)
        # print(img2)

for i in os.listdir('dataset/'):
    file = os.path.join('dataset/', i)
    file = file + '/'
    count = 0
    for v in os.listdir(file):
        img = os.path.join(file, v)
        # print(img)
            # print(img)
        print(img)
        name = str(count) + '_after' +'.jpg'
        name = os.path.join(file, name)
        # print(name)
        face_crop(img, img, name)
        count += 1
    print(file, count)

# img = cv2.imread('./dataset/with_mask/with_mask162.jpg')
# cv2.imshow("Img", img)
# cv2.waitKey()
# face_crop('./dataset/with_mask/with_mask162.jpg', './dataset/with_mask/with_mask162.jpg', './dataset/with_mask/111111.jpg')