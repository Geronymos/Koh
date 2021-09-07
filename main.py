#!/usr/bin/env python

import cv2
import mediapipe as mp
import sys
import csv
import math
from functools import reduce 
from itertools import chain
import pyvirtualcam

from components.FeatureSwitcher import FeatureSwitcher
from components.loader import read_dataset

running = True

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# define color of landmarks
drawing_spec_webcam = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
drawing_spec_match = mp_drawing.DrawingSpec(thickness=1, circle_radius=1,color=(0,0,255))

feature_switcher = FeatureSwitcher()

def match(face_mesh, webcam_image, dataset):

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

  features = feature_switcher.get_combined()
  
  if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:

      important_webcam = [face_landmarks.landmark[important[0]] for important in features["show"]]
        # [*chain([face["triplets"][important[0]] for important in features])]

      face_vertices = []
      for landmark in important_webcam:
        face_vertices.append(landmark.x)
        face_vertices.append(landmark.y)
        face_vertices.append(landmark.z)

      sorted_images = sorted(dataset, key=lambda elem: math.dist(face_vertices, elem["important"]))
      top_image = sorted_images[0]["filename"]
      # print(top_image)
      print(top_image, len(dataset[0]["important"]), len(face_vertices))

      image_match = cv2.imread(top_image)

      mp_drawing.draw_landmarks(
          image=webcam_image,
          landmark_list=sorted_images[0]["landmarks"],
          connections=feature_switcher.get_combined()["match"],
          landmark_drawing_spec=drawing_spec_match,
          connection_drawing_spec=drawing_spec_match)

      cv2.imshow('Selected image', image_match)

      # show webcam image with wireframe
      mp_drawing.draw_landmarks(
          image=webcam_image,
          landmark_list=face_landmarks,
          connections=features["match"],
          landmark_drawing_spec=drawing_spec_webcam,
          connection_drawing_spec=drawing_spec_webcam)

  # display the state of which featues should be matched 
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


def main():

  # get dataset file path as first parameter
  dataset_filename = sys.argv[1]

  webcam = cv2.VideoCapture(0)

  cv2.namedWindow('Selected image', cv2.WINDOW_NORMAL)
  cv2.resizeWindow('Selected image', 800, 600)

  dataset = read_dataset(dataset_filename)

  """ track webcam face  """
  with mp_face_mesh.FaceMesh( max_num_faces=1, min_detection_confidence=0.5 ) as face_mesh:
    while webcam.isOpened() and running:
      success, image = webcam.read()
      if success: 
        match(face_mesh, image, dataset)
        feature_switcher.key_events(dataset, running)
      else:
        # If loading a video, use 'break' instead of 'continue'.
        print("Ignoring empty camera frame.")
        continue
  
  webcam.release()

if __name__ == "__main__":
    main()
