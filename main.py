#!/usr/bin/env python

import cv2
import mediapipe as mp
import sys
import csv
import json
import math

import face_components

# landmark_pb2.NormalizedLandmarkList

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

trained_file = sys.argv[1]

# important_landmarks = [face[important[0]] for important in connectionsSwitcher.get_combined()["show"]]

class Koh:
  def __init__(self, images_faces):
    self.data = images_faces

    self.drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    self.webcam = cv2.VideoCapture(0)

    self.connectionsSwitcher = face_components.ConnectionsSwitcher()

    self.running = True

    cv2.namedWindow('Selected image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Selected image', 800, 600)
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        # min_tracking_confidence=0.5
        ) as self.face_mesh:
      while self.webcam.isOpened() and self.running:
        success, image = self.webcam.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue

        self.render(image)
        self.key_events()
    
    self.webcam.release()

  def match_faces(self, vertices_list1, vertices_list2):
    return math.dist(vertices_list1, vertices_list2)

  def vertices_list_from_face(self, face):
    vertices_list = [[landmark.x, landmark.y, landmark.z] for landmark in face]
    vertices_list = [item for sublist in vertices_list for item in sublist]

    return vertices_list

  def render(self, webcam_image):


    # flip
    webcam_image = cv2.cvtColor(cv2.flip(webcam_image, 1), cv2.COLOR_BGR2RGB)
    webcam_image.flags.writeable = False

    # track face
    results = self.face_mesh.process(webcam_image)

    webcam_image.flags.writeable = True
    webcam_image = cv2.cvtColor(webcam_image, cv2.COLOR_RGB2BGR)

    height, width, _color = webcam_image.shape

    if results.multi_face_landmarks:
      for dataset_face_landmarks in results.multi_face_landmarks:
        
        face_landmarks_list = [landmark for landmark in dataset_face_landmarks.landmark]

        
        important_landmarks_dataset = []
        important_landmarks_webcam = []
        for important in self.connectionsSwitcher.get_combined()["match"]:
          important_landmarks_dataset.append(face_landmarks_list[ important[0] ])
          important_landmarks_webcam.extend(self.data[ important[0]*3 : important[0]*3+2 ])

        face_xyz = self.vertices_list_from_face([landmark for landmark in face_landmarks_list])


        sorted_images = sorted(self.data, key=lambda elem: self.match_faces(elem["data"], face_xyz))
        top_image = sorted_images[0]["file"]
        print(top_image)

        image_match = cv2.imread(top_image)

    cv2.imshow('Selected image', image_match)

    for i, key in enumerate(self.connectionsSwitcher.states):
      cv2.putText(
        webcam_image,
        f"{key}: {self.connectionsSwitcher.states[key]}",
        (width - 100, 10 * (i+1)),
        cv2.FONT_HERSHEY_SIMPLEX,
        .3, 
        (255, 255, 255, 255),
        1
      )

    mp_drawing.draw_landmarks(
        image=webcam_image,
        landmark_list=dataset_face_landmarks,
        connections=self.connectionsSwitcher.get_combined()["match"],
        landmark_drawing_spec=self.drawing_spec,
        connection_drawing_spec=self.drawing_spec)

    cv2.imshow('MediaPipe FaceMesh', webcam_image)

  def key_events(self):
    key = cv2.waitKey(1) & 0xFF
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


images_faces = []

with open(trained_file, "r") as file:
  # images_faces = json.load(file)
  csv_reader = csv.reader(file)

  for image_file, *positions in csv_reader:
    # csv.process(row)
    images_faces.append({
      "file": image_file,
      "data": [float(position) for position in positions]
    })

many_faces = Koh(images_faces)