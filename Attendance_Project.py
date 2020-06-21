import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

#access the folder with images

path = 'Images'                                             #the path where the registered images are stored
mylist = os.listdir(path)                                   #getting a list of the filenames of registered images

#Step 0 -> register images

images=[]                                                   # a list for images
classnames = []                                             # a list for names

for cl in mylist:
    curimg = cv2.imread(f'{path}/{cl}')
    images.append(curimg)                                   # add new image to the list
    classnames.append(os.path.splitext(cl)[0])              # add new name to the list


#Step 1 ->function to get the encodings of all the images in the folder (all registered people)
def find_encodings(images):
    encode_list=[]                                          # an empty list for storing the encodings of all the faces
    for img in images:
        img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)            # iterate through every image and change it from BGR TO RGB
        encode = face_recognition.face_encodings(img)[0]    # find the face and get all the encodings
        encode_list.append(encode)                          # add the found encodings to the list
    print('Images Encoded')
    return encode_list

#Step 4  -> Function for marking the attendance
def markAttendance(name):
    with open('Attendance.csv','r+') as f:                  # open the csv file where attendance data is stored
        namelist = []                                       # a list of the names already in the file
        myDataList = f.readlines()
        for line in myDataList:
            entry= line.split(',')
            namelist.append(entry[0])
        if name not in namelist:                            # a check so that there is no discrepancy in the file
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')             # get the current data in HourMinSec format in string data type
            f.writelines(f'\n{name},{dtString}')            # mark the attendance for the new face detected

encodelistKnown = find_encodings(images)
#Step 2 recognize the face
cap = cv2.VideoCapture(0)                                   # get the input, in this case from webcam
while True:
    success,img = cap.read()                                # get the images of each frames
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)             # resize the image to 0.25 of its size for faster computations
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    faceLoc = face_recognition.face_locations(imgS)         # detect the face with help of face recognition library and store its location
    encodes = face_recognition.face_encodings(imgS,faceLoc) # also get the encodings found at the face location


    for encode,face in zip(encodes,faceLoc):
        matches = face_recognition.compare_faces(encodelistKnown,encode)    # match the current image frame with the already registered faces
        facedis = face_recognition.face_distance(encodelistKnown,encode)    # get the distance(difference) of each face from registered faces

        matchindex = np.argmin(facedis)                     # get the argument of the registered face with minimum difference from the current face
        if matches[matchindex]:
            name=classnames[matchindex].upper()             # convert the name to uppercase
            y1,x2,y2,x1 = face
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4               # re-scale the face to its original size

            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)  # draw a rectangle around the face
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0))  # display the name at the bottom

            markAttendance(name)    # mark the attendance of the detected face

    cv2.imshow('Webcam',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.destroyAllWindows()
        break



