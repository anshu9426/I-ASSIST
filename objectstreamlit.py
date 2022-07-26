from numpy.core.numeric import normalize_axis_tuple
import streamlit as st
import numpy as np
from PIL import Image
import cv2 
import mediapipe as mp

import numpy as np
import cv2
from gtts import gTTS
from playsound import playsound
import os

import cv2
import face_recognition
import numpy as np
from face_recognition.api import face_encodings


# mp_drawing = mp.solutions.drawing_utils
# mp_face_mesh = mp.solutions.face_mesh
# model_face_mesh=mp_face_mesh.FaceMesh()
st.title("Image filtering web Application Developed By-Aditya Agarwal")
st.subheader("Image Operation")
st.write("This application performs various image conversion opertaion with the help of openCV,numpy,streamlit,pillow library")


add_selectbox=st.sidebar.selectbox(
    "What operation would you like to perform?",
    ("About","Grayscale Filter","Blue Filter","Green Filter","Face Meshing")
)

if add_selectbox=="About":
    st.write("An Appliaction that add filters to your image")

if add_selectbox=="Grayscale Filter":
    # st.write("Image Converted to Grayscale")
    # image_file_path=st.sidebar.file_uploader("Upload image")
    # if image_file_path is not None:
    #     image=np.array(Image.open(image_file_path))
    #     st.sidebar.image(image)
    #     gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #     st.image(gray_image)
    thres = 0.5  # Threshold to detect object
    nms_threshold = 0.2  # (0.1 to 1) 1 means no suppress , 0.1 means high suppress
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH,280) #width
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT,120) #height
    # cap.set(cv2.CAP_PROP_BRIGHTNESS,150) #brightness

    classNames = []
    with open('coco.names', 'r') as f:
        classNames = f.read().splitlines()
    print(classNames)

    font = cv2.FONT_HERSHEY_PLAIN
    #font = cv2.FONT_HERSHEY_COMPLEX
    Colors = np.random.uniform(0, 255, size=(len(classNames), 3))

    weightsPath = "frozen_inference_graph.pb"
    configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    while True:
        success, img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=thres)
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1, -1)[0])
        confs = list(map(float, confs))
        #print(type(confs[0]))
        #print(confs)

        indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)
        if len(classIds) != 0:
            for i in indices:
                i = i[0]
                box = bbox[i]
                confidence = str(round(confs[i], 2))
                color = Colors[classIds[i][0]-1]
                x, y, w, h = box[0], box[1], box[2], box[3]
                cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness=2)
                cv2.putText(img, classNames[classIds[i][0]-1]+" "+confidence, (x+10, y+20),
                            font, 1, color, 2)
                # speak_Text(classNames[classIds[i][0]-1])
                # cv2.putText(img,str(round(confidence,2)),(box[0]+100,box[1]+30),
                # font,1,colors[classId-1],2)
                print(classNames[classIds[i][0]-1])

        k = cv2.waitKey(1)
        if k == ord('q'):
            cv2.destroyAllWindows()
            break
        # speak_Text(classNames[classIds[i][0]-1])
        cv2.imshow("Output", img)
        # cap.release()
        # cv2.waitKey(1)
# elif add_selectbox == "Blue Filter":
#     image_file_path = st.sidebar.file_uploader("Upload image")
#     if image_file_path is not None:
#         image = np.array(Image.open(image_file_path))
#         st.sidebar.image(image)
#         zeros=np.zeros(image.shape[:2],dtype="uint8")
#         r,g,b=cv2.split(image)

#         blue_image = cv2.merge([zeros,zeros,b])
#         st.image(blue_image)


elif add_selectbox=="Face Meshing":

    chris = face_recognition.load_image_file('chris.png')
    chris_encodings = face_recognition.face_encodings(chris)[0]

    robert = face_recognition.load_image_file('robert.png')
    robert_encodings = face_recognition.face_encodings(robert)[0]

    known_face_encodings = [chris_encodings, robert_encodings]
    known_face_names = ["Chris", "Robert"]

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            print("Colud not access the camera")
            break
        small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding)
            name = "UNKNOWN"
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)

            print(face_names)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left+6, bottom-6),
                        font, 1.0, (255, 255, 255), 1)

        cv2.imshow("Frame", cv2.cvtColor(rgb_small_frame, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


        
        
            
            
                    



