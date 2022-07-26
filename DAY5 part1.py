#image classification falls under the Supervised learning 
import cv2
import face_recognition

image_train=face_recognition.load_image_file('chris.png')
image_encodings=face_recognition.face_encodings(image_train)[0]

#to draw a box on the image location
image_locations_train=face_recognition.face_locations(image_train)[0]


image_test=face_recognition.load_image_file('robert.png')
image_encodings_test=face_recognition.face_encodings(image_test)[0]

results=face_recognition.compare_faces([image_encodings],image_encodings_test)[0]
dst=face_recognition.face_distance([image_encodings],image_encodings_test)
print(results)
if results:
    image_train=cv2.cvtColor(image_train,cv2.COLOR_BGR2RGB)
    cv2.rectangle(image_train,(image_locations_train[3],image_locations_train[0]),(image_locations_train[1],image_locations_train[2]),(0,255,0),2)
    cv2.putText(image_train,f"{results}{dst}",(60,60),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)
    cv2.imshow("Chris",image_train)
else:
    image_train=cv2.cvtColor(image_train,cv2.COLOR_BGR2RGB)
    cv2.rectangle(image_train,(image_locations_train[3],image_locations_train[0]),(image_locations_train[1],image_locations_train[2]),(0,0,255),2)
    cv2.putText(image_train, f"{results}{dst}", (60, 60),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 2)
    cv2.imshow("Chris",image_train)

cv2.waitKey(0)

