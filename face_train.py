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
    for (x, y, w, h) in faces:
        f= image[y:y+w,x:x+h]
        cv2.imwrite('temp.jpg',f)
        im='temp.jpg'
        if recognizer:
            nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
            predicted_name=g[nbr_predicted-1]
            #print "{} is Correctly Recognized with confidence {}".format(predicted_name, conf)
            if conf>=140:
                continue
            msg='Is this '+predicted_name
            reply = buttonbox(msg, image=im, choices=['Yes','No'])
            if reply=='Yes':
                reply=predicted_name
                directory=current_directory+'\\'+reply
                if not os.path.exists(directory):
                    os.makedirs(directory)
                random_name=''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(7))
                path=directory+'\\'+random_name+'.jpg'
                cv2.imwrite(path,f)
            else:
                msg = "Who is this?"
                reply = buttonbox(msg, image=im, choices=choices)
                if reply == 'Add a name':
                    name=enterbox(msg='Enter the name', title='Training', strip=True)
                    #print name
                    choices+=[name]
                    reply=name
                directory=current_directory+'\\'+reply
                if not os.path.exists(directory):
                    os.makedirs(directory)
                random_name=''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(7))
                path=directory+'\\'+random_name+'.jpg'
                #print path
            cv2.imwrite(path,f)
        else:
            msg = "Who is this?"
            reply = buttonbox(msg, image=im, choices=choices)
            if reply == 'Add a name':
                name=enterbox(msg='Enter the name', title='Training', strip=True)
                #print name
                choices+=[name]
                reply=name
            directory=current_directory+'\\'+reply
            if not os.path.exists(directory):
                os.makedirs(directory)
            random_name=''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(7))
            path=directory+'\\'+random_name+'.jpg'
            #print path
            cv2.imwrite(path,f)
        os.remove(im)




# calculate window position
root = Tk()
pos = int(root.winfo_screenwidth() * 0.5), int(root.winfo_screenheight() * 0.2)
root.withdraw()
WindowPosition = "+%d+%d" % pos

# patch rootWindowPosition
rootWindowPosition = WindowPosition
            
def train(img,recognizer):
    print 'Currently processing '+img
    imagePath = current_directory+'\\'+img
    get_name(imagePath,recognizer)
    

cascPath = 'haarcascade_frontalface_default.xml' #Face detection xml file
os.system("change_name.py") #Read change_name.py. It renames all the photos in all the folders sequentially.
for filename in os.listdir("."):
    os.system("change_name.py")
    recognizer=''
    try:
        recognizer=train_recognizer()
    except:
        recognizer=False
    imagePath=filename
    train(imagePath,recognizer)
    os.remove(filename)
    print 'Done with this photograph'
    

        
