#!/usr/bin/env python

import cv2
import mediapipe as mp
import sys
import csv
import json
import math
from functools import reduce 

from components.FeatureSwitcher   import FeatureSwitcher 
from components.DatasetController import DatasetController
from components.FaceConverter     import FaceConverter
from components.LandmarkConverter import LandmarkConverter
from components.TrackingHelper    import TrackingHelper


# landmark_pb2.NormalizedLandmarkList

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

dataset_filename = sys.argv[1]

class Koh:
  def __init__(self, filename):
    self.dataset = self.load_from_file(filename)
    self.featureSwitcher = FeatureSwitcher()

    self.running = True

    self.drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    self.webcam = cv2.VideoCapture(0)

    cv2.namedWindow('Selected image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Selected image', 800, 600)
    
    with mp_face_mesh.FaceMesh( max_num_faces=1, min_detection_confidence=0.5 ) as self.face_mesh:
      while self.webcam.isOpened() and self.running:
        success, image = self.webcam.read()
        if success: 
          self.main(image)
          self.key_events()
        else:
          # If loading a video, use 'break' instead of 'continue'.
          print("Ignoring empty camera frame.")
          continue
    
    self.webcam.release()

  def load_from_file(self, filename):

    dataset_faces = []

    with open(filename, "r") as file:
      csv_reader = csv.reader(file)

      for image_file, *positions in csv_reader:
        face_converter = FaceConverter(image_name=image_file)
        face_converter.from_vertices_list([float(position) for position in positions])
        dataset_faces.append(face_converter)

    return DatasetController(dataset_faces)

  def main(self, webcam_image):
    track = TrackingHelper(self.face_mesh, webcam_image)

    matches = self.dataset.match(track.faces[0], self.featureSwitcher.get_combined()["show"])

    # for webcam_face in track.faces:

    #   webcam_face = webcam_face.to_face()
    #   face_xyz = webcam_face.to_vertices_list()

    #   sorted_images = sorted(self.dataset, key=lambda elem: self.match_faces(elem["data"], face_xyz))
    #   top_image = sorted_images[0]["file"]
    #   print(top_image)

    #   image_match = cv2.imread(top_image)

  # def render(self, webcam_image):
    if len(matches) != 0:
      cv2.imshow('Selected image', cv2.imread(matches[0].image))

    for i, [key, value] in enumerate(self.featureSwitcher.states.items()):
      cv2.putText(
        track.image,
        f"{key}: {value}",
        (track.width - 100, 10 * (i+1)),
        cv2.FONT_HERSHEY_SIMPLEX,
        .3, 
        (255, 255, 255, 255),
        1
      )

    # mp_drawing.draw_landmarks(
    #     image=track.image,
    #     landmark_list=track.faces[0].to_face(),
    #     connections=self.featureSwitcher.get_combined()["match"],
    #     landmark_drawing_spec=self.drawing_spec,
    #     connection_drawing_spec=self.drawing_spec)

    cv2.imshow('MediaPipe FaceMesh', track.image)

  def key_events(self):
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
    # self.connectionsSwitcher.switch_state()

    if key is ord("q"): 
      self.running = False
    elif key is ord("a"): # all
      self.connectionsSwitcher.switch_state("every")
    elif key is ord("f"): # face
      self.connectionsSwitcher.switch_state("face")
    elif key is ord("o"): # oval
      self.connectionsSwitcher.switch_state("face_oval")
    elif key is ord("m"): # mouth/lips
      self.connectionsSwitcher.switch_state("lips")
    elif key is ord("r"): # right eye
      self.connectionsSwitcher.switch_state("right_eye")
    elif key is ord("y"): # left eyebrow
      self.connectionsSwitcher.switch_state("left_eyebrow")
    elif key is ord("l"): # left eye
      self.connectionsSwitcher.switch_state("left_eye")
    elif key is ord("x"): # right eyebrow
      self.connectionsSwitcher.switch_state("right_eyebrow")

many_faces = Koh(dataset_filename)