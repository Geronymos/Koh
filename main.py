#!/usr/bin/env python

import cv2
import mediapipe as mp
import sys
import csv
import json
import math
from functools import reduce 

from components.FeatureSwitcher   import FeatureSwitcher 
# from components.DatasetController import DatasetController
# from components.FaceConverter     import FaceConverter
# from components.LandmarkConverter import LandmarkConverter
# from components.TrackingHelper    import TrackingHelper


# landmark_pb2.NormalizedLandmarkList

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

dataset_filename = sys.argv[1]

feature_switcher = FeatureSwitcher()

running = True

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
webcam = cv2.VideoCapture(0)

cv2.namedWindow('Selected image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Selected image', 800, 600)


dataset_faces = []

with open(dataset_filename, "r") as file:
  csv_reader = csv.reader(file)

  dataset = []

  for image_file, *positions in csv_reader:
    
    positions = [float(position) for position in positions]

    landmarks = []

    for x,y,z in zip(*(iter(positions),) * 3):
      normalized_landmark = mp.framework.formats.landmark_pb2.NormalizedLandmark()
      normalized_landmark.x = x
      normalized_landmark.y = y
      normalized_landmark.z = z
      landmarks.append(normalized_landmark)

    dataset.append({
      "filename": image_file,
      "landmarks": landmarks,
      "vertices": positions
    })


def main(webcam_image):

  width, height, _color = webcam_image.shape
  # Flip the image horizontally for a later selfie-view display, and convert
  # the BGR image to RGB.
  webcam_image = cv2.cvtColor(cv2.flip(webcam_image, 1), cv2.COLOR_BGR2RGB)
  # To improve performance, optionally mark the image as not writeable to
  # pass by reference.
  webcam_image.flags.writeable = False
  results = face_mesh.process(webcam_image)

  # Draw the face mesh annotations on the image.
  webcam_image.flags.writeable = True
  webcam_image = cv2.cvtColor(webcam_image, cv2.COLOR_RGB2BGR)
  if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:

      face_vertices = []
      for landmark in face_landmarks.landmark:
        face_vertices.append(landmark.x)
        face_vertices.append(landmark.y)
        face_vertices.append(landmark.z)

      sorted_images = sorted(dataset, key=lambda elem: math.dist(face_vertices, elem["vertices"]))
      top_image = sorted_images[0]["filename"]
      print(top_image)

      image_match = cv2.imread(top_image)
      cv2.imshow('Selected image', image_match)

      # show webcam image with wireframe
      mp_drawing.draw_landmarks(
          image=webcam_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACE_CONNECTIONS,
          landmark_drawing_spec=drawing_spec,
          connection_drawing_spec=drawing_spec)

  for i, [key, value] in enumerate(feature_switcher.states.items()):
    cv2.putText(
      webcam_image,
      f"{key}: {value}",
      (width - 100, 10 * (i+1)),
      cv2.FONT_HERSHEY_SIMPLEX,
      .3, 
      (255, 255, 255, 255),
      1
    )

  cv2.imshow('MediaPipe FaceMesh', webcam_image)

def key_events():
  key_action = {
    "a": "ever",
    "f": "face",
    "o": "face_oval",
    "m": "lips",
    "r": "right_eye",
    "y": "left_eyebrow",
    "l": "left_eye",
    "x": "right_eyebrow",
  }

  key = cv2.waitKey(1) & 0xFF

  # if key 
  # self.feature_switcher.switch_state()

  if key is ord("q"): 
    running = False
  elif key is ord("a"): # all
    feature_switcher.switch_state("every")
  elif key is ord("f"): # face
    feature_switcher.switch_state("face")
  elif key is ord("o"): # oval
    feature_switcher.switch_state("face_oval")
  elif key is ord("m"): # mouth/lips
    feature_switcher.switch_state("lips")
  elif key is ord("r"): # right eye
    feature_switcher.switch_state("right_eye")
  elif key is ord("y"): # left eyebrow
    feature_switcher.switch_state("left_eyebrow")
  elif key is ord("l"): # left eye
    feature_switcher.switch_state("left_eye")
  elif key is ord("x"): # right eyebrow
    feature_switcher.switch_state("right_eyebrow")

with mp_face_mesh.FaceMesh( max_num_faces=1, min_detection_confidence=0.5 ) as face_mesh:
  while webcam.isOpened() and running:
    success, image = webcam.read()
    if success: 
      main(image)
      key_events()
    else:
      # If loading a video, use 'break' instead of 'continue'.
      print("Ignoring empty camera frame.")
      continue

webcam.release()