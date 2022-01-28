import face_recognition
from cv2 import cv2
import numpy as np
import os
from datetime import datetime, timedelta
import pyttsx3
# from vosk import Model, KaldiRecognizer
# import pyaudio
#
# model = Model(r'C:\Users\David Erivona\Documents\Face_Reg\model large')
# recognizer = KaldiRecognizer(model, 16000)
# speak = pyttsx3.init()
#
# cap =pyaudio.PyAudio()
# stream = cap.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True,
# frames_per_buffer=8192)
# stream.start_stream()
#
# while True:
#     # engine = pyttsx3.init()
#     data = stream.read(4096, exception_on_overflow=False)
#
#     if recognizer.AcceptWaveform(data):
#         result = recognizer.Result()
#         result = literal_eval(result)

path = 'img'
images = []
class_names = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    class_names.append(os.path.splitext(cl)[0])
print(class_names)

def find_encodings(imgages):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('project.csv','r+') as f:
        myDataList = f.readlines()
        # print(myDataList)
        name_list = []
        for line in myDataList:
            entry = line.split(',')
            name_list.append(entry[0])
        if name not in name_list:
            now = datetime.now()
            time = now.strftime("%H:%M:%S")
            date = now.strftime("%Y-%m-%d")
            f.writelines(f"\n{name},{time},{date}")
            new_date = datetime.today() + timedelta(days=1)
            date_new = new_date.strftime("%Y-%m-%d")
            if date == date_new:
                f.writelines(f"\n{name},{time},{date_new}")

encodeListKnown = find_encodings(images)
# print(len(encodeListKnown))
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_small = cv2.resize(img,(0,0),None,0.25,0.25)
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    faceCur = face_recognition.face_locations(img_small)
    encodeCur = face_recognition.face_encodings(img_small, faceCur)

    for encodeFace,faceLoc in zip(encodeCur,faceCur):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)

        matchIndex = np.argmin(faceDis)
        value = faceDis[matchIndex]
        if value<0.4:
            if matches[matchIndex]:
                name = class_names[matchIndex].upper()
                # print(name)
                y1,x2,y2,x1 = faceLoc
                y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,name,(x1+6,y2+6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)
                # engine.say(f'Hi {name}')
                # engine.runAndWait()
        elif value>=0.4:
            if matches[matchIndex]:
                name = 'Stranger Unknown'
                name = name.upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 + 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                # engine.say('Hello There I do not know you')
                # engine.runAndWait()

    cv2.imshow('Face Recognition Project',img)
    cv2.waitKey(1)
