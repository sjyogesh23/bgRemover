import cv2
import glob
import os

imagefiles = glob.glob("Images/*")
imagefiles.sort()

images = []
for i in imagefiles:
    img = cv2.imread(i)
    if img is not None:
        images.append(img)

stitcher = cv2.Stitcher_create()
status, result = stitcher.stitch(images)

if status == 0:
    output_folder = 'output'
    os.makedirs(output_folder, exist_ok=True)

    output_filename = os.path.join(output_folder, 'output_image.jpg')
    cv2.imwrite(output_filename, result)
    
    cv2.imshow('image',result)
    cv2.waitKey(0)
cv2.destroyAllWindows()