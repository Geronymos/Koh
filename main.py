#!/usr/bin/env python

import cv2
import sys

from components.loader import load
from components.track import Tracking
from components.match import match
from components.keyEvents import key_events
from components.draw import Drawing
from components.FeatureSwitcher   import FeatureSwitcher 

dataset_filename = sys.argv[1]

# feature_switcher = FeatureSwitcher()
tracking = Tracking()
drawing = Drawing()

running = True


dataset = load(dataset_filename)

# def calc_important_landmarks()

webcam = cv2.VideoCapture(0)

# while webcam.isOpened() and running:
while True:
    success, image = webcam.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if not success: 
      # If loading a video, use 'break' instead of 'continue'.
      print("Ignoring empty camera frame.")
      continue

    image = cv2.flip(image,1)

    #state = key_events()
    #if state.close:
    #    break
    face = tracking.track(image)
    top_image = match(face, dataset)
    print(top_image["image"])
    drawing.drawFace(face[0], image)
    drawing.drawMatch(top_image, image)
    drawing.drawWebcam(image)
    #drawing.drawStates(state, webcam)


webcam.release()
cv2.destroyAllWindows()
