import cv2
import os
import numpy as np

def faceDetection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    haar_face = cv2.CascadeClassifier(r"/Users/francescamartinucci/Desktop/Principi-Face_Recognition/haarcascade_frontalface_default.xml")
    face = haar_face.detectMultiScale(gray, scaleFactor=1.3,minNeighbors=6)

    return face, gray
    
def training_data(directory):
    faces=[]
    facesID=[]
    for path, subdir, filename in os.walk(directory):
        for filename in filename:
            if filename.startswith("."):
                print("skipping system file")
                continue
            id = os.path.basename(path)
            image_path=os.path.join(path, filename)
            img_test = cv2.imread(image_path)
            if img_test is None:
                print("error opening image")
                continue
            face, gray = faceDetection(img_test)
            if len(face)!= 1: #verifico che sia restituita 1 sola immagine (faccia?)
                continue
            (x,y,w,h) = face[0]
            region = gray[y:y+w, x:x+h]
            print(region)
            faces.append(region) #per ogni faccia si salva la regione della foto in cui ha riscontrato la presenza di un volto
            facesID.append(int(id))
            print(faces)
            print(facesID)
        return faces, facesID
    

def prova(directory):
    faces=[]
    facesID=[]
    for path, subdir, filenames in os.walk(directory):
        print(path, subdir, filenames)
        for filename in filenames:
            if filename.startswith("."):
                print("skipping system file")
                continue
            # print(filename)
            id = os.path.basename(path)
            # print("sottocartella a cui appartiene la foto(0 o 1 elon o johnny)",id)
            image_path=os.path.join(path, filename) #percorso completo della foto
            # print(image_path)
            img_test = cv2.imread(image_path) #lettura immagine
            # print(img_test)
            if img_test is None:
                print("error opening image", image_path)
                continue
            face, gray = faceDetection(img_test)
            if len(face)!= 1: #verifico che sia restituita 1 sola immagine (faccia?)
                continue
            # print("restituita 1 sola immagine")
            (x,y,w,h) = face[0]
            region = gray[y:y+w, x:x+h]
            # print(region)
            faces.append(region) #per ogni faccia si salva la regione della foto in cui ha riscontrato la presenza di un volto
            # print(faces)
            facesID.append(int(id))
            # print(facesID)

    return faces, facesID

# prova("/Users/francescamartinucci/Desktop/Principi-Face_Recognition/immagini")

def train_classifier(faces, facesID):

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    # print(type(facesID[0]))
    face_recognizer.train(faces, np.array(facesID))
    face_recognizer.write('Principi-Face_Recognition/trainedModel.yml')
    return face_recognizer

def draw_rect(test_img,face):
	(x,y,w,h)=face
	cv2.rectangle(test_img, (x,y), (x+w, y+h), (0,0,255), thickness=5)

def put_name(test_img, text, x,y):
	cv2.putText(test_img, text,(x,y), cv2.QT_FONT_NORMAL, 2, (0,0,255), 3)

print("hello")

#https://www.youtube.com/watch?v=fodFa4mDaEQ - Riconoscimento dei Volti
#https://www.youtube.com/watch?v=Oaz5ooilVRw - in Tempo Reale
#https://www.youtube.com/watch?v=GFjfTrpaDXk - Come Creare un Filtro Instagram
