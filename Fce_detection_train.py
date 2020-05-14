import cv2
import numpy as np
import os

cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("C:/Users/verma/Desktop/OpenCv/haar.xml")  # xml with face proportions

name = input("Enter your name: ")
count = int(input("No. of pics: "))
images = []

while True:
    retval, image = cap.read()
    if retval:
        faces = classifier.detectMultiScale(image)
        cv2.imshow("cropped", image)
        if len(faces) > 0:
            sorted_faces = sorted(faces, key=lambda item: item[2] * item[3])

            face1 = sorted_faces[-1]

            x, y, w, h = face1

            cut_1 = image[y:y + h+100, x:x + w+100]
            cut_1 = cv2.resize(cut_1, (100, 100))
            cv2.imshow("cropped", cut_1)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    if key == ord('c'):
        while count >= 0:
            img = cv2.cvtColor(cut_1, cv2.COLOR_BGR2GRAY)
            images.append(img.flatten())
            count -= 1
            print(count)

cap.release()
cv2.destroyAllWindows()

X = np.array(images)
y = np.full((X.shape[0], 1), name)

data = np.hstack([y, X])  # first column has name of whose image is in the other rest of the columns (just like in
# knn) 784 features of image

if os.path.exists("faces.npy"): #to update the file with new image data.
    old = np.load("faces.npy")
    data = np.vstack([old, data])

np.save("faces.npy", data)
