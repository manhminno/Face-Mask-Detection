import cv2
import argparse

from PIL import Image
from utils import *

def predict(img_detect, model):
    img_detect = cv2.resize(img_detect, (32, 32)) #Resize 32x32
    img = Image.fromarray(img_detect)     
    
    img = data_transform(img)    
    img = img.view(1, 3, 32, 32) #View in tensor
    img = Variable(img)      
    
    model.eval() #Set eval mode

    #To Cuda
    model = model.cuda()
    img = img.cuda()

    output = model(img)
    
    predicted = torch.argmax(output)
    p = label2id[predicted.item()]

    return  predicted




model = CNN()
model = model.cuda()
model.load_state_dict(torch.load('weights/Face-Mask-Model.pt'))

img = cv2.imread('./dataset/without_mask/0_0_dongchengpeng_0005')
# a = predict(img, model)
# print(a)

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades  + 'haarcascade_frontalface_default.xml')
# #Detect face
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# for (x,y,w,h) in faces:
#     img2 = img[y+2:y+h-2, x+2:x+w-2]
#     emo = predict(img2, model)  #face index 
#     # face = label2id[emo.item()]
#     print(emo)
#     # putface(img, face, x, y, w, h)
result = predict(img, model)
print(result)
cv2.imshow("Img", img)
cv2.waitKey()