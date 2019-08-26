import cv2
from scipy import ndimage
import numpy as np
import math
import pytesseract


#scaling factors
scalePercent=70; #reduces the dimension of the image
rscale=100/scalePercent; #brings back to the original dimension


#scale the image to make computations faster
def scale(image):
    im=cv2.imread(image)
    width = int(im.shape[1] * scalePercent/ 100)
    height = int(im.shape[0] * scalePercent/ 100)
    dim = (width, height)
    im2 = cv2.resize(im, dim, interpolation = cv2.INTER_AREA) 
    return im2
    
#GrayScale conversion 
def preprocess(image):
    imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return imgray

#Canny Edge filter and Contour detection
def detect(image,org):
    cv2.imshow("GrayScale Image", image)
    edged = cv2.Canny(image, 170, 200)
    cv2.imshow("Edged Number Plate", edged)
    contours,hierarchy=cv2.findContours(edged,1,2)
    tempContours=sorted(contours, key = cv2.contourArea, reverse=True)
    fContours=[]
    for cnt in tempContours:
        aspect_ratio=0
        r=cv2.boundingRect(cnt)
        x,y,width,height = r
        if(width!=0 and height!=0):
            aspect_ratio = min(width, height) / max(width, height)
            if(aspect_ratio>0.21 and aspect_ratio<0.25):
                fContours.append(cnt)
    c=sorted(fContours,key=cv2.contourArea,reverse=True)[0]
    x,y,w,h = cv2.boundingRect(c)
    new_img = org[y:y+h, x:x+w]
    cv2.imwrite("cropped.jpg", new_img)
    cv2.imshow("Croppped Number Plate", new_img)
    cv2.rectangle(org,(x,y),(x+w,y+h),(0,255,0),2)
    config = ('-l eng --oem 1 --psm 3')
    text = pytesseract.image_to_string(new_img, config=config)
    print(text)
    cv2.imshow("Detetcted Number Plate", org)
    
    cv2.waitKey(0)
    
    
detect(preprocess(scale('car4.jpg')),scale('car4.jpg'))