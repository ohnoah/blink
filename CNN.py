# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import csv
import cv2
import matplotlib.pyplot as plt


def eye_aspect_ratio(eye):
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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())

# datapoints to remember EAR
EAR = []
Label = []
Time = []
i = 0
ear = 0.0

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
# if using Webcam
# vs = FileVideoStream(args["video"]).start()
fileStream = True
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
# fileStream = False
time.sleep(1.0)

# loop over frames from the video stream
while True:
    # if this is a file video stream, then we need to check if
    # there any more frames left in the buffer to process
    # if fileStream and not vs.more():
    #     break

    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    Time.append(i)
    i=i+1
    frame = vs.read()
    # test if video has stopped playing
    if frame is None:
        break
    frame = imutils.resize(frame, width=960)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        (a,b) = (100000,0)
        (c,d) = (0,10000)
        for (x,y) in leftEye:
            (a,b) = min(x,a),max(y,b)
            (c,d) = max(c,x), min(d,y)
        rightEye = shape[rStart:rEnd]
        (e,f) = 10000, 0 
        (g,h) = 0,10000
        for (x,y) in rightEye:
            (e,f) = min(x,e),max(y,f)
            (g,h) = max(g,x), min(h,y)
        cv2.circle(frame, (a,b), 5, (0,255,0))
        cv2.circle(frame, (c,d), 5, (0,255,0))
        cv2.circle(frame, (e,f), 5, (0,255,0))
        cv2.circle(frame, (g,h), 5, (0,255,0))
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

    # save datapoint
    EAR.append(ear)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
    if key == ord('p'):
        Label.append(0.5)
    else:
        Label.append(0.0)

        # if the `q` key was pressed, break from the loop
    if key == ord('q'):
        print("Broke")
        break

    #cv2.imwrite("./pizza/frame%d.jpg" % i, frame)

#Save file
outfile1 = open(str(args["video"])+"_label" + '.csv','w')
writer=csv.writer(outfile1)
writer.writerows(map(lambda x: [x], Label))
outfile1.close()

plt.plot(Time, EAR)
plt.plot(Time, Label)
plt.savefig(str(args["video"]) + '.png')

outfile2 = open(str(args["video"])+"_ear" + '.csv','w')
writer=csv.writer(outfile2)
writer.writerows(map(lambda x: [x], EAR))
outfile2.close()

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()