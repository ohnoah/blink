import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pywt
from hmmlearn import hmm
from scipy.spatial import distance as dist
import pygame, time, sys, os
from random import randint
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
from scipy.spatial import distance as dist

class BlinkDetector:
    def __init__(self, shape_predictor_file, batch_interval):
        self.fileStream = True
        self.EAR = []
        self.prevEAR = []
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor_file)
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.batchInterval = batch_interval
        self.timeElapsed = 0
        self.prevTime = -1
        self.blinkR = 0

    def eye_aspect_ratio(self, eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])

        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        # return the eye aspect ratio
        return ear

    def blinkrate(self):
        if(len(self.EAR) > 3):
            print("NON-EMPTY EAR")
            self.EAR = self.prevEAR
        else:
            print("EMPTY EAR")
        model = hmm.GaussianHMM(n_components=2)
        model.fit(np.array(self.EAR).reshape(-1,1))
        states = model.predict(np.array(self.EAR).reshape(-1,1))
        count = 0
        prevstate = 1
        for i in range(len(states)):
            if states[i] == 0 and prevstate == 1:
                count += 1
                prevstate = 0
            elif states[i] == 1 and prevstate == 0:
                prevstate = 1
        self.EAR.clear()
        print("THIS IS THE COUNT" + str(count))
        return count

    def processFrame(self, vs):
        if self.prevTime == -1:
            self.prevTime = time.time()
        elif self.timeElapsed < self.batchInterval:
            newTime = time.time()
            self.timeElapsed += newTime - self.prevTime
            self.prevTime = newTime
        else:
            self.timeElapsed = 0
            if(len(self.EAR) > 3):
                self.prevEAR = self.EAR
            ##CLEARS EAR
            self.blinkR = self.blinkrate()
            print("HI BATCH TIME" + str(self.blinkR))
            self.prevTime = time.time()

        frame = vs.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)

            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # save datapoint
            self.EAR.append(ear)
        returnVal = 0
        if(self.blinkR < 3):
            returnVal = -2
        elif(3 <= self.blinkR <= 6 ):
            returnVal = -1
        elif(7 < self.blinkR <= 11 ):
            returnVal = 0
        elif(11 < self.blinkR <= 15 ):
            returnVal = 1
        else:
            returnVal = 2
            
        return returnVal
 
class Background:
    def __init__(self, imageFilePath, width, height):
        self.background = pygame.transform.scale(pygame.image.load(imageFilePath),(width,height))
        self.width = width
        self.height = height
        #scrolling position
        self.xpos = width
    def updateBackground(self, screen):
        if self.xpos < 0:
            self.xpos = self.width
        screen.blit(self.background, (self.xpos,0))
        screen.blit(self.background, (self.xpos-self.width,0))
        self.xpos -= self.width*0.001
    def resize(self,w,h):
        self.xpos *= w/self.width
        self.width = w
        self.height = h
        self.background = pygame.transform.scale(self.background, (w,h))

charwidth = 100
charheight = 50
averageBlinkR = 0
def main():
    # Initialise video stream
    vs = VideoStream(src=0).start()
    bd = BlinkDetector("shape_predictor_68_face_landmarks.dat", 10)
    not_done = True
    pygame.init()
    print("DO WE GET HERE")
    character1 = pygame.transform.scale(pygame.image.load("./rocket.png"), (charwidth,charheight))
    character = pygame.transform.rotate(character1, 1)
    screen = pygame.display.set_mode((900,300), pygame.RESIZABLE)
    bg = Background("./background.jpeg", 900,300)
    
    try:
        counter = 1
        charY = bg.height/2
        while not_done:
            blinkR = bd.processFrame(vs)
            if((counter % 15) == 0):
                charY = bg.height/2 + (bg.height/10)*blinkR
                print(blinkR)
                if(charY > bg.height - charheight/2):
                    charY = bg.height - charheight/2
                elif(charY < charheight/2):
                    charY = charheight/2
                    
                counter = 0
            pygame.display.update()
            arr = pygame.event.get()
            #deals with closing and resizing
            for event in arr:
                if event.type == pygame.QUIT:
                    print("ITS MEANT TO CLOSE")
                    not_done = False
                    break
                if event.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode((event.w,event.h), pygame.RESIZABLE)
                    bg.resize(event.w, event.h)
            bg.updateBackground(screen)

            (x,y) = ((bg.width-charwidth)/2,charY - charheight/2)
            screen.blit(character,(x,y))
            #done drawing so sleep
            counter += 1
            pygame.time.wait(33)
    except Exception as e: 
        print(e)
        not_done = False
    print("HELLO")
    pygame.quit()
    time.sleep(3)
    os._exit(0)
    
main()
