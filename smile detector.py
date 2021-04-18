import cv2
print ("whats with that smile")
train_face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_detector = cv2.CascadeClassifier("haarcascade_smile.xml")

#grab video cam feed
webcam = cv2.VideoCapture(0)


#show current webcam frame
while True:
    succesful_webcam_read,frame = webcam.read()

    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detect faces and smile
    detecting_faces = train_face_detector.detectMultiScale(grayscaled_frame)
    #detecting_smile = smile_detector.detectMultiScale(grayscaled_frame,scaleFactor=1.75,minNeighbors=20)

    for (x, y, w, h) in detecting_faces:
        #drawing rectangle around faces
        cv2.rectangle(frame, (x, y), (x + w, x + h), (0 ,0,255), 5)

        #getting face subframe
        the_face: object = frame[ y:y+h, x:x+w]

        grayscaled_face = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)



#find all smiles within face
        detecting_smile = smile_detector.detectMultiScale(grayscaled_face, scaleFactor=1.75, minNeighbors=20)
        for (x_, y_, w_, h_) in detecting_smile:
            cv2.rectangle(the_face, (x_, y_), (x_ + w_, y_ + h_), (0, 255, 0), 5)

        #labelling face
        if len(detecting_smile)>0:
            cv2.putText(frame,"smiling",(x,y+h+40),fontScale=3,fontFace= cv2.FONT_HERSHEY_COMPLEX,color=(255,0,255))

    cv2.imshow("detected_face", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()

#