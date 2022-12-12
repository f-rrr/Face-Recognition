#videocapture.py
import os
import cv2
import numpy as np
import facerecognition as fr

faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceRecognizer.read(r"Principi-Face_Recognition/trainedModel.yml")

name = nomi = {0:"Vincenzo", 1:"Vittorio", 2:"Francesca"}

capture = cv2.VideoCapture(0)

while True:
	ret, test_img=capture.read() #ret = bool, test_img: frame dalla telecamera
	faces_detected, gray = fr.faceDetection(test_img)

	#for (x,y,w,h) in faces_detected:
	#	cv2.rectangle(test_img, (x,y), (x+w,y+h), (0,0,255), thickness=3)

	for face in faces_detected:
		(x,y,w,h) = face
		region = gray[y:y+w, x:x+h]
		label, confidence = faceRecognizer.predict(region)
		print("conficence", confidence)
		predictedName = name[label]
		print("name: ", predictedName)
		fr.draw_rect(test_img, face)
		if confidence > 42:
			resized = cv2.resize(test_img,(1920,1080))
			cv2.imshow("face", resized)
			cv2.waitKey(10)
			continue
		fr.put_name(test_img, predictedName,x,y)
		
		resized = cv2.resize(test_img,(1920,1080))
		cv2.imshow("face", resized)
		cv2.waitKey(15)