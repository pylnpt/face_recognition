import cv2

#Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
pre_trained_face_data = cv2.CascadeClassifier('haarcascade_frontal_face.xml')

#Choose an image to detect faces in and grayscale it
detectable_img = cv2.imread('34.webp')
grayscaled_img = cv2.cvtColor(detectable_img, cv2.COLOR_BGR2GRAY)

#Detect faces
face_coordinates = pre_trained_face_data.detectMultiScale(grayscaled_img)

#Draw a rectangle to a face
for (x, y, w, z) in face_coordinates:
    cv2.rectangle(detectable_img,  (x, y), (x + w, y + z), (0, 255, 0), 2)

print(face_coordinates)

#Show the face
cv2.imshow('Face detector', detectable_img)
cv2.waitKey();

print("Code completed")