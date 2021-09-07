#!/usr/bin/env python

import cv2
import mediapipe as mp
import sys
import math
import pyvirtualcam
import numpy as np

from components.FeatureSwitcher import FeatureSwitcher
from components.loader import read_dataset

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

"""  define color of landmarks """
drawing_spec_webcam = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
drawing_spec_match = mp_drawing.DrawingSpec(thickness=1, circle_radius=1,color=(0,0,255))

feature_switcher = FeatureSwitcher()

def match(face_mesh, webcam_image, dataset):

  """ To improve performance, optionally mark the image as not writeable to
  pass by reference. """
  webcam_image.flags.writeable = False
  results = face_mesh.process(webcam_image)

  """ Draw the face mesh annotations on the image. """
  webcam_image.flags.writeable = True
  webcam_image = cv2.cvtColor(webcam_image, cv2.COLOR_RGB2BGR)

  features = feature_switcher.get_combined()
  
  if not results.multi_face_landmarks:
    return (False, 0,0)

  for face_landmarks in results.multi_face_landmarks:

    """ get important landmarks for webcam """
    important_webcam = [face_landmarks.landmark[important[0]] for important in features["show"]]

    face_vertices = []
    for landmark in important_webcam:
      face_vertices.append(landmark.x)
      face_vertices.append(landmark.y)
      face_vertices.append(landmark.z)

    """ get top match """

    sorted_images = sorted(dataset, key=lambda elem: math.dist(face_vertices, elem["important"]))
    top_image = sorted_images[0]

    print(top_image["filename"], len(dataset[0]["important"]), len(face_vertices))

    return (True, top_image, face_landmarks)

def main():

  """ get dataset file path as first parameter """
  dataset_filename = sys.argv[1]

  webcam = cv2.VideoCapture(0)

  """ create windows """

  cv2.namedWindow('Selected image', cv2.WINDOW_NORMAL)
  cv2.resizeWindow('Selected image', 800, 600)

  dataset = read_dataset(dataset_filename)

  """ setup virtual camera """

  fmt = pyvirtualcam.PixelFormat.BGR
  with pyvirtualcam.Camera(width=1280, height=720, fps=20, fmt=fmt) as cam:
    print(f'Using virtual camera: {cam.device}')
    frame = np.zeros((cam.height, cam.width, 3), np.uint8)  # RGB
    
    """ track webcam face  """

    with mp_face_mesh.FaceMesh( max_num_faces=1, min_detection_confidence=0.5 ) as face_mesh:
      while webcam.isOpened() and feature_switcher.states["running"]:
        success, image = webcam.read()

        if not success:
          """  If loading a video, use 'break' instead of 'continue'. """
          print("Ignoring empty camera frame.")
          continue 

        """ Flip the image horizontally for a later selfie-view display, and convert
        the BGR image to RGB. """
        webcam_image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        width, height, _color = webcam_image.shape

        """ mark important landmarks in dataset """

        features = feature_switcher.key_events()
        for face in dataset:
          face["important"] = []
          for important in features:
            face["important"].extend(face["triplets"][important[0]])

        success, top_image, face_landmarks = match(face_mesh, webcam_image, dataset)

        if not success:
          continue

        image_match = cv2.imread(top_image["filename"])

        """ draw landmarks """

        mp_drawing.draw_landmarks(
            image=webcam_image,
            landmark_list=top_image["landmarks"],
            connections=feature_switcher.get_combined()["match"],
            landmark_drawing_spec=drawing_spec_match,
            connection_drawing_spec=drawing_spec_match)

        cv2.imshow('Selected image', image_match)

        # show webcam image with wireframe
        mp_drawing.draw_landmarks(
            image=webcam_image,
            landmark_list=face_landmarks,
            connections=feature_switcher.get_combined()["match"],
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

        """ send virtual webcam frame """

        frame = cv2.resize(image_match, (cam.width, cam.height), interpolation=cv2.BORDER_DEFAULT)
        cam.send(frame)
        cam.sleep_until_next_frame()
  
  webcam.release()

if __name__ == "__main__":
    main()
