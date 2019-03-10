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
import math

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
        self.vs = VideoStream(src=0).start()

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
        data = np.array(self.EAR)
        if data.size==0:
            print("datasize")
            return 0
        for i in range(0, data.size, math.ceil(data.size*0.05)):
            lim = math.ceil(data.size*0.05)
            if lim + i > data.size:
                lim = data.size - i
            data[i:i+lim] -= np.mean(data[i:i+lim])
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

    def processFrame(self):
        if self.prevTime == -1:
            self.prevTime = time.time()
        elif self.timeElapsed < self.batchInterval:
            newTime = time.time()
            self.timeElapsed += newTime - self.prevTime
            self.prevTime = newTime
        else:
            self.timeElapsed = 0
            if(len(self.EAR) < 3):
                self.EAR = self.prevEAR
            ##CLEARS EAR
            self.blinkR = self.blinkrate()
            print("HI BATCH TIME" + str(self.blinkR))
            self.prevTime = time.time()

        frame = self.vs.read()
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
        if(len(self.EAR) > 3):
            self.prevEAR = self.EAR
        returnVal = 0
        if(self.blinkR < 6):
            returnVal = -2
        elif(6 <= self.blinkR <= 12 ):
            returnVal = -1
        elif(12 < self.blinkR <= 18 ):
            returnVal = 0
        elif(18 < self.blinkR <= 24 ):
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

class Exhaust:
    def __init__(self, imageFilePath, width, height, xpos, ypos):
        self.image = pygame.transform.scale(pygame.image.load(imageFilePath),(width,height))
        self.x = xpos
        self.y = ypos

class Bar:
    def __init__(self, screen):
        #HARCODED
        #HARDCODED
        
        self.HEAT_BAR_IMAGE = pygame.Surface((0.8*(screen.get_width()), 20))
        self.color = pygame.Color(240, 240, 240)
        self.heat = 0.0
        for x in range(self.HEAT_BAR_IMAGE.get_width()):
            for y in range(self.HEAT_BAR_IMAGE.get_height()):
                self.HEAT_BAR_IMAGE.set_at((x, y), self.color)
        pygame.font.init()
        myfont = pygame.font.SysFont('Comic Sans MS', 30)
        self.textsurface = myfont.render('0', False, (240, 240, 240))

    def update(self, counter, screen, bg):
        heightbar = 0.95*bg.height
        left_bar = 0.1*bg.width
        heat_rect = self.HEAT_BAR_IMAGE.get_rect(bottomleft=(left_bar, heightbar - 20))
        # `heat` is the percentage of the surface's width and
        # is used to calculate the visible area of the image.
        self.heat = self.heat + counter / 30  # 5% of the image are already visible.
        screen.blit(
            self.HEAT_BAR_IMAGE,
            heat_rect,
            # Pass a rect or tuple as the `area` argument.
            # Use the `heat` percentage to calculate the current width.
            (0, 0, heat_rect.w / 100 * self.heat, heat_rect.h)
        )
        myfont = pygame.font.SysFont('Comic Sans MS', 30)
        five_mins= myfont.render('5 mins', False, (240, 240, 240))
        ten_mins= myfont.render('10 mins', False, (240, 240, 240))
        fifteen_mins= myfont.render('15 mins', False, (240, 240, 240))
        twenty_mins= myfont.render('20 mins', False, (240, 240, 240))
        screen.blit(self.textsurface, (left_bar, heightbar))
        if self.heat>25:
            screen.blit(five_mins, (left_bar + heat_rect.w/100 * 25, heightbar))
        if self.heat>50:
            screen.blit(ten_mins, (left_bar + heat_rect.w/100 * 50, heightbar))
        if self.heat>75:
            screen.blit(fifteen_mins, (left_bar + heat_rect.w/100 * 75, heightbar))
        if self.heat>100:
            screen.blit(twenty_mins, (left_bar + heat_rect.w, heightbar))
        pygame.display.flip()

    def resize(self,w,h):
        self.width = w
        self.height = h
        self.background = pygame.transform.scale(self.HEAT_BAR_IMAGE, (w,h))

class Score:
    def __init__(self, s_w, s_h):
        pygame.font.init()
        self.score = 0
        self.s_w = s_w
        self.s_h = s_h
        self.count = 0
        self.change = 0

    def displayScore(self, screen):
        if self.count==0:
            myfont = pygame.font.SysFont('Score:' + str(self.score), 45)
            score_surface = myfont.render('Score: ' + str(self.score), False, (240, 240, 240))
            screen.blit(score_surface, (self.s_w, self.s_h))
        else:
            myfont = pygame.font.SysFont(str(self.change), 45)
            if self.change == -5:
                score_surface = myfont.render(str(self.change), False, (255, 0, 0))
            if self.change == -1:
                score_surface = myfont.render(str(self.change), False, (240, 150, 150))
            if self.change == 5:
                score_surface = myfont.render("+" + str(self.change), False, (44, 200, 100))
            screen.blit(score_surface, (self.s_w, self.s_h))
            self.count = self.count - 1

    def updateScore(self, screen, val):
        self.change = self.valtoScore(val)
        self.score = self.score + self.change
        self.count = 5

    def valtoScore(self, val):
        if val==-2 | val==2:
            return -5
        if val==-1 | val==-1:
            return -1
        elif val==0:
            return 5


charwidth = 100
charheight = 50
averageBlinkR = 0
def main():
    # Initialise video stream
    bd = BlinkDetector("shape_predictor_68_face_landmarks.dat", 10)
    not_done = True
    pygame.init()
    character1 = pygame.transform.scale(pygame.image.load("./rocket.png"), (charwidth,charheight))
    character = pygame.transform.rotate(character1, 1)
    screen = pygame.display.set_mode((1024,512), pygame.RESIZABLE)
    bg = Background("./copy.jpeg", 1024,512)
    clouds = []
    ytargets = [0.2, 0.35, 0.5, 0.65, 0.8]
    transitionstep = 0
    score_position_w = 0.1*bg.width
    score_position_h = 0.2*bg.height
    bar = Bar(screen)
    score = Score(score_position_w, score_position_h)

    try:
        counter = 1
        charY = bg.height/2
        prevTimeElapsed = 0
        while not_done:
            blinkR = bd.processFrame()
            if((counter % 15) == 0):
                counter = 0
            # Update exhaust and ship position
            if prevTimeElapsed > bd.timeElapsed:
                clouds.append(Exhaust("exhaust.png", 30, 15, (bg.width-charwidth)/4 * 3, charY - charheight/2 + 15))
                transitionstep = ((ytargets[blinkR + 2] * bg.height) - charY) / 100
                score.updateScore(screen, blinkR)
            charY += transitionstep
            if (transitionstep < 0 and charY < ytargets[blinkR + 2] * bg.height) or (transitionstep > 0 and charY > ytargets[blinkR + 2] * bg.height):
                transitionstep = 0
            prevTimeElapsed = bd.timeElapsed
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
            (x,y) = ((bg.width-charwidth)/4 * 3,charY - charheight/2)
            screen.blit(character,(x,y))
            for cloud in clouds:
                cloud.x -= bg.width * 0.0001
                screen.blit(cloud.image, (cloud.x, cloud.y))
            bar.update(counter, screen, bg)
            score.displayScore(screen)
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
