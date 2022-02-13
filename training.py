import numpy as np
import cv2
import os

import face_recognition as fr

test_img = cv2.imread(r'C:\Users\Dhairya Veera\Desktop\CODING\synapse\alz\test3.png')
# cv2.imshow("test image",test_img)

faces_detected, gray_img = fr.faceDetection(test_img)
# cv2.imshow("gray img", gray_img)
# cv2.waitKey(0)
print("face detected: ", faces_detected)

faces,faceID=fr.labels_for_training_data(r'C:\Users\Dhairya Veera\Desktop\CODING\synapse\alz\images') #Give path to the train-images folder which has both labeled folder as 0 and 1
face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer.save(r'C:\Users\Dhairya Veera\Desktop\CODING\synapse\alz\trainingData.yml') #It will save the trained model. Just give path to where you want to save

name={0:"Dhairya", 1:"preksha"}    #Change names accordingly. If you want to recognize only one person then write:- name={0:"name"} thats all. Dont write for id number 1.


# for face in faces_detected:
#     (x,y,w,h)=face
#     roi_gray=gray_img[y:y+h,x:x+h]
#     label,confidence=face_recognizer.predict(roi_gray)
#     print ("Confidence :",confidence)
#     print("label :",label)
#     fr.draw_rect(test_img,face)
#     predicted_name=name[label]
#     fr.put_text(test_img,predicted_name,x,y)

# resized_img=cv2.resize(test_img,(500,500))

# cv2.imshow("face detection ", resized_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows