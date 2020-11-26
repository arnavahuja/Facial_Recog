# Facial_Recog

## Introduction
This project was undertaken as a part of an Internship at TamilNadu Health Sytem Reforms. Facial Recognition has far reaching uses in the Healthcare Industry. In this current situation of the pandemic, methods such as biometric are a huge risk to spreading infection. Facial Recognition software can be an efficient solution to reduce the risk of infection spreading.
Facial recognition is one of the best ways to perform mass identification as it does not require the cooperation of the test subject to work. Properly designed systems installed in airports, multiplexes, and other public places can identify individuals among the crowd, without passers-by even being aware of the system. This makes it fit for our purpose of queueing up patients in large hospitals to save time.

## Problem
Here we focused on using facial recognition to solve the given problem. The structure of the model will be:
  1) Separating patients and non-patients.
  2) Further dividing patients into registered and unregistered categories.
  3) Registration of new patients and assigning token numbers to old ones, to put them in a queue.
  4) Calling patients for consultation based on the token numbers assigned.

The first step will involve separating the people into two categories. The patients and non-patients will walk in separate channels. The model will be used on patients. Their faces will be identified in real time and matched with a database of faces.
In the second step, faces of patients which were not recognised will be marked as new registrations.
For the third step, they will be sent to a registration desk, where their details will be taken and mapped to their face. The new registration will be stored in the database and the database will be updated. If the face is recognised, the patient will be sent a token number on their registered phone number.
After this, they will be sent to a waiting room, where a list of token numbers/ names of the patients
will be displayed.
Finally, in step four, based on that list, patients will be called one by one for consultation.
  
## Approach/Methodology Used

The model will use a Haar Cascade algorithm to carry out facial recognition. It is a machine learning based approach where a cascade function is trained from a lot of
positive and negative images. It is then used to detect objects in other images.
The algorithm has four stages:
  1. Haar Feature Selection
  2. Creating Integral Images
  3. Adaboost Training
  4. Cascading Classifiers

## Result
Real time facial recognition can significantly reduce the size of queues, reduce patient wait time and speed up the system, paving way for overall modernisation of the system.
And in the era of the pandemic, such a software is needed. 
The haar cascade algorithm has a high accuracy of above 95%.

  

