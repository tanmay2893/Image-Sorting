import cv2
import sys
import os,random,string
import os
current_directory=os.path.dirname(os.path.abspath(__file__))
from Tkinter import Tk
from easygui import *
import numpy as np
x= os.listdir(current_directory)
new_x=[]
testing=[]
for i in x:
    if i.find('.')==-1:
        new_x+=[i]
    else:
        testing+=[i]
x=new_x
try:
    x.remove('Renamed Phtographs')
except:
    pass
g=x
choices=['Add a name']+x
y= range(1,len(x)+1)

def get_images_and_labels():
    global current_directory,x,y,g
    if x==[]:
        return (False,False)
    image_paths=[]
    for i in g:
        path=current_directory+'\\'+i
        for filename in os.listdir(path):
            final_path=path+'\\'+filename
            image_paths+=[final_path]
    images = []
    labels = []
    for image_path in image_paths:
        #print image_path
        # Read the image and convert to grayscale
        img = cv2.imread(image_path,0)
        # Convert the image format into numpy array
        image = np.array(img, 'uint8')
        # Get the label of the image
        backslash=image_path.rindex('\\')
        underscore=image_path.index('_',backslash)
        nbr = image_path[backslash+1:underscore]
        t=g.index(nbr)
        nbr=y[t]
        # If face is detected, append the face to images and the label to labels
        images.append(image)
        labels.append(nbr)
        #cv2.imshow("Adding faces to traning set...", image)
        #cv2.waitKey(50)
    # return the images list and labels list
    return images, labels
# Perform the tranining
def train_recognizer():
    recognizer = cv2.createLBPHFaceRecognizer()
    images, labels = get_images_and_labels()
    #print images
    #print labels
    if images==False:
        return False
    cv2.destroyAllWindows()
    recognizer.train(images, np.array(labels))
    #print recognizer
    return recognizer

def get_name(image_path,recognizer):
    global x,choices
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    x1=testing
    global g
    image = cv2.imread(image_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    predict_image = np.array(img, 'uint8')
    faces = faceCascade.detectMultiScale(
        img,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    final_name=''
    all_names=[]
    for (x, y, w, h) in faces:
        f= image[y:y+w,x:x+h]
        nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
        predicted_name=g[nbr_predicted-1]
        #print "{} is Correctly Recognized with confidence {}".format(predicted_name, conf)
        if conf>=140:
            continue
        reply=predicted_name

        if predicted_name=='Random':
            if not (predicted_name in all_names):
                all_names.append(predicted_name)
                final_name+='Others&'
        else:
            final_name+=predicted_name
            final_name+='&'
    print final_name[:-1]
    return final_name[:-1]



def rename(img,recognizer):
    from shutil import copyfile
    print 'Currently processing '+img
    imagePath = current_directory+'\\'+img
    final_name = get_name(imagePath,recognizer)+'.jpg'
    #new_path = current_directory+'\\'+'Renamed Photographs'
    #final_name = new_path+'\\'+get_name(imagePath,recognizer)+'.jpg'
    #if not os.path.exists(new_path):
    #    os.makedirs(new_path)
    #copyfile(imagePath,final_name)
    i=1
    while os.path.exists(final_name):
        final_name=final_name[:-4]+str(i)+'.jpg'
        i+=1
    os.rename(img,final_name)

cascPath = 'haarcascade_frontalface_default.xml' #Face detection xml file
for filename in os.listdir("."):
    if filename.endswith('.jpg'):
        os.system("change_name.py")
        recognizer=''
        try:
            recognizer=train_recognizer()
        except:
            recognizer=False
        imagePath=filename
        rename(imagePath,recognizer)
        #os.remove(filename)
        print 'Done with this photograph'
