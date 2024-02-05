import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 420)
seg = SelfiSegmentation()


allimgs = os.listdir("Images")
imglist = []
for imgpath in allimgs:
    img = cv2.imread(f'Images/{imgpath}')
    imglist.append(img)
print("""
        1. Press A - Previous image
        2. Press D - Next Image
        3. Press Q - Quit
    """) 
imgindex = 0

while True:
    success, img = cap.read()
        
    if len(imglist) > 0:
        imgbg = cv2.resize(imglist[imgindex], (img.shape[1], img.shape[0]))
        imgout = seg.removeBG(img, imgbg, cutThreshold=0.5)
    else:
        imgout = seg.removeBG(img, (255,255,255), cutThreshold=0.7)
    
    
    opscreen = cvzone.stackImages([img, imgout],2,1)
    cv2.imshow("Image", opscreen)
    
    key = cv2.waitKey(1)
    
    if key == ord('a'):
        if imgindex>0:
            imgindex -= 1
        else:
            imgindex = len(imglist) - 1
    if key == ord('d'):
        if imgindex<len(imglist) - 1:
            imgindex += 1
        else:
            imgindex = 0
    elif key == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()