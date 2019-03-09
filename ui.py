import pygame, time, sys, os
from random import randint

def blinkrate():
    
    return randint(-10,10)


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
    not_done = True
    pygame.init()
    screen = pygame.display.set_mode((900,300), pygame.RESIZABLE)
    bg = Background("background.jpeg", 900,300)
    character1 = pygame.transform.scale(pygame.image.load("rocket.png"), (charwidth,charheight))
    character = pygame.transform.rotate(character1, 1)

    
    #get the blink rate 
    #BLINKKKK
    
    try:
        counter = 1
        charY = bg.height/2
        blinkR = averageBlinkR
        while not_done:
            if((counter % 10) == 0):
                print("updating y")
                blinkR += blinkrate()/100
                charY = bg.height/2 + (bg.height/2)*blinkR
                if((abs(charY - bg.height) < charheight/2)):
                    charY = bg.height - charheight/2
                elif(charY < bg.height/2):
                    charY < bg.height/2
                    
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
            screen.blit(character,((bg.width-charwidth)/2,charY + charheight/2))
            #done drawing so sleep
            counter += 1
            pygame.time.wait(33)
    except Exception as e: 
        print(e)
        not_done = False
    print("HELLO")
    pygame.quit()
    os._exit(0)
    
main()
