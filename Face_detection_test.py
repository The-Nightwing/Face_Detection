import cv2
import numpy as np
import os

from sklearn.neighbors import KNeighborsClassifier

cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("C:/Users/verma/Desktop/OpenCv/haar.xml")  # xml with face proportions
data = np.load("faces.npy")

X = data[:, 1:]
y = data[:, 0]

model = KNeighborsClassifier()
model.fit(X, y)


while True:
    retval, image = cap.read()
    if retval:
        faces = classifier.detectMultiScale(image)
        if len(faces)>0:
            for face in faces:
                x,y,w,h=face

                cut = image[y:y + h, x:x + w]

                resized = cv2.resize(cut, (100, 100))

                y_test = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) # because the face_detect_train has trained photos
                # with gray scale
                y_test = y_test.flatten()

                output = model.predict([y_test])[0] # list of outputs are there so only one has to be the title of
                # rectangle

                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 3)
                cv2.putText(image, str(output), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

        cv2.imshow("face_Recog", image)

        key = cv2.waitKey(1)

        if key == ord('q'):
            break
cap.release()
cv2.destroyWindow()