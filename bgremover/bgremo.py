import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
cap = cv2.VideoCapture(0)
cap.set(3, 728)
cap.set(4, 410)
seg = SelfiSegmentation()


allimgs = os.listdir("Images")
imglist = []
for imgpath in allimgs:
    img = cv2.imread(f'Images/{imgpath}')
    imglist.append(img)
    
imgindex = 0

while True:
    success, img = cap.read()
    
    imgbg = cv2.resize(imglist[imgindex], (img.shape[1], img.shape[0]))
    imgout = seg.removeBG(img, imgbg , cutThreshold=.7)
    
    opscreen = cvzone.stackImages([img, imgout],2,1)
    cv2.imshow("Image", opscreen)
    key = cv2.waitKey(1)
    if key == ord('a'):
        if imgindex>0:
            imgindex -= 1
        else:
            imgindex = len(imglist) - 1
    elif key == ord('d'):
        if imgindex<len(imglist) - 1:
            imgindex += 1
        else:
            imgindex = 0
    elif key == ord('q'):
        break
    print(imgindex)