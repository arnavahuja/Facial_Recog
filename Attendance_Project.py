import cv2
import numpy as np
import face_recognition
import os

#access the folder with images

path = 'Images'
mylist = os.listdir(path)

#register images

images=[]
classnames = []

for cl in mylist:
    curimg = cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classnames.append(os.path.splitext(cl)[0])


#function to get the encodings of all the images in the folder (all registered people)
def find_encodings(images):
    encode_list=[]
    for img in images:
        img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    print('Images Encoded')
    return encode_list

encodelistKnown = find_encodings(images)
#Step 2 recognize the face
cap = cv2.VideoCapture(0)
while True:
    success,img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    faceLoc = face_recognition.face_locations(imgS)
    encodes = face_recognition.face_encodings(imgS,faceLoc)


    for encode,face in zip(encodes,faceLoc):
        matches = face_recognition.compare_faces(encodelistKnown,encode)
        facedis = face_recognition.face_distance(encodelistKnown,encode)

        matchindex = np.argmin(facedis)
        if matches[matchindex]:
            name=classnames[matchindex].upper()
            y1,x2,y2,x1 = face
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4

            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0))
    cv2.imshow('Webcam',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.destroyAllWindows()
        break



