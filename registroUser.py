#-*- coding: utf-8 -*-
import cv2
import os

if not os.path.exists('DataSet'):
	print('Directorio creada: DataSet')
	os.makedirs('DataSet')
else:
    print('Carpeta encontrada!')   

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

count = 0
while True:
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    key = cv2.waitKey(1) & 0xFF

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(128,0,255),2)
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150), interpolation=cv2.INTER_CUBIC)
        if key == ord('g'):
            cv2.imwrite('DataSet/rostro_{}.jpg'.format(count),rostro)
            count = count +1
    cv2.rectangle(frame,(10,5),(450,25),(255,255,255),-1)
    cv2.putText(frame,'Presione g, para tomar fotos - Presione q, para salir',
        (10,20), 2, 0.5,(255,0,0),1,cv2.LINE_AA)
    cv2.imshow('Registro de Usuarios',frame)

    if key == ord('q'):
        break
print('Se realizo el registro de {} fotos'.format(count))
cap.release()
cv2.destroyAllWindows()