#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 11:10:22 2021

@author: bouchelliga_hedi
"""

import cv2
import mediapipe as mp
import time
current_time = 0
previous_time = 0
mpHANDS = mp.solutions.hands
hands = mpHANDS.Hands()
mpDRAW = mp.solutions.drawing_utils


## read video

cap = cv2.VideoCapture(0)


while True:
    
    
    suc,frame = cap.read()

    imgargb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = hands.process(imgargb)
    
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handmark in results.multi_hand_landmarks:
      ##      mpDRAW.draw_landmarks(frame,handmark)  ## draw the keypoints in the hand
            mpDRAW.draw_landmarks(frame,handmark,mpHANDS.HAND_CONNECTIONS) ## draw keypoints lines
            for id,lm in enumerate(handmark.landmark):
                print(id,lm)
    current_time = time.time()
    fps = 1/(current_time-previous_time)
    previous_time = current_time
    text_fps = "FPS= "
    cv2.putText(frame,text_fps,(10,60),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0))
    cv2.putText(frame,str(round(fps)),(100,60),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0))
    ##cv2.imshow("KEYPOINTS",frame)
    cv2.imshow("HAND POINTS CONNECTION",frame)
    cv2.waitKey(10)