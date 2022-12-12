import cv2
import os
import numpy as np
import facerecognition as fr

test_img = cv2.imread(r"/Users/francescamartinucci/Downloads/valerio.jpeg")
dim = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

face, gray = fr.faceDetection(test_img)
# for (x,y,w,h) in face:
#     cv2.rectangle(test_img, (x,y), (x+w,y+h), (0,0,255), thickness=5)

# faces, facesID = fr.training_data('/Users/francescamartinucci/Desktop/Principi-Face_Recognition/immagini')
# print(faces, facesID)
# faceRecognizer = fr.train_classifier(faces, facesID)
# faceRecognizer.save("trainedData.yml")


faces, facesID= fr.prova("Principi-Face_Recognition/immagini")
# print("f",faces, facesID)
faceRecognizer = fr.train_classifier(faces, facesID)

# faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
# faceRecognizer.read(r"/Users/francescamartinucci/Desktop/Principi-Face_Recognition/trainedData.yml")

#creo vettore dei nomi
names = {0 : "Vincenzo", 1 : "Vittorio", 2 : "Francesca"}

if len(face) == 0:
	print("Faccia non trovata")
for face in face:
	(x,y,w,h) = face
	region = gray[y:y+h, x:x+w]
	label, confidence = faceRecognizer.predict(region)
 	#modello, prego a lei, identifichi la foto
	print("confidence", confidence)
	print("label", label)
	print("predizione", names[label])
	fr.draw_rect(test_img, face)
	predicted_name = names[label]
	if(confidence>25.5): #+ è grade + è probabile che si sia sbagliato
		print("Impostore!")
		continue
	fr.put_name(test_img, predicted_name, x,y)

resized= cv2.resize(test_img, (dim.shape[1],dim.shape[0]))
cv2.imshow("faccia trovata?", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()