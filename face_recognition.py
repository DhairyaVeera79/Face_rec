from cgi import test
import numpy as np
import cv2
import os

def faceDetection(test_img):
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    face_haar =cv2.CascadeClassifier(r'C:\Anaconda\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml')
    faces = face_haar.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=3)
    return faces, gray_img

def labels_for_training_data(directory):
    faces=[]
    faceID=[]

    for path, subdirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("skipping system file")
                continue
            id=os.path.basename(path)
            img_path=os.path.join(path, filename)
            print("img_path: ", img_path)
            print("id: ", id)
            test_img = cv2.imread(img_path)
            if test_img is None:
                print("not loaded properly")
                continue

            faces_rect, gray_img = faceDetection(test_img)
            (x,y,w,h)=faces_rect[0]
            roi_gray = gray_img[y:y+w, x:x+h]
            faces.append(roi_gray)
            faceID.append(int(id))
    return faces, faceID

def train_classifier(faces, faceId):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(faceId))
    return face_recognizer

def draw_rect(test_img, face):
    (x,y,w,h) = face
    cv2.rectangle(test_img, (x,y), (x+w, y+h), (0,255,0), thickness=3)

def put_text(test_img, label_name, label_rel, label_age, label_loc, x, y):
    text_size, _ = cv2.getTextSize("aaa", cv2.FONT_HERSHEY_DUPLEX, 1, 3)
    line_height = text_size[1] + 5
    cv2.putText(test_img, label_name, (x,y-(3*line_height)), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 3)
    cv2.putText(test_img, label_rel, (x,y-(2*line_height)), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 3)
    cv2.putText(test_img, label_age, (x,y-line_height), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 3)
    cv2.putText(test_img, label_loc, (x,y), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 3) 
    
    
    
    


             


    
