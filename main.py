import cv2
import mediapipe as mp
import json
import math

import face_components

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

trained_file = "dataset_thispersondoesnotexist.json"
trained_file = "dataset_friends.json"
trained_file = "dataset_jim_carrey.json"
trained_file = "dataset_partyAllTheTime.json"
trained_file = "dataset_kaggle.json"
trained_file = "dataset_vgg_face_dataset_images.json"
trained_file = "dataset_rickroll.json"

mp_face_mesh.FACE_CONNECTIONS

def xyz_list_from_face(face):
  # important_landmarks = [face[important[0]] for important in mp_face_mesh.FACE_CONNECTIONS]
  important_landmarks = [face[important[0]] for important in face_components.components["lips"]]
  xyz_list = [[landmark["x"], landmark["y"], landmark["z"]] for landmark in important_landmarks]
  xyz_list = [item for sublist in xyz_list for item in sublist]


  return xyz_list

def match_faces(xyz_face1, xyz_face2):
  return math.dist(xyz_face1, xyz_face2)

with open(trained_file, "r") as file:
  images_faces = json.load(file)

  for image in images_faces:
    xyz_list = xyz_list_from_face(image["faces"][0])

    image["xyz"] = xyz_list

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    # min_tracking_confidence=0.5
    ) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    height, width, color = image.shape

    cv2.namedWindow('Selected image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Selected image', 800, 600)

    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:

        face_dict = [{"x": landmark.x, "y": landmark.y, "z": landmark.z} for landmark in face_landmarks.landmark]

        # print(face_dict)

        face_xyz = xyz_list_from_face(face_dict)

        sorted_images = sorted(images_faces, key=lambda elem: match_faces(elem["xyz"], face_xyz))
        top_image = sorted_images[0]["file"]
        print(top_image)

        image_match = cv2.imread(top_image)

        cv2.imshow('Selected image', image_match)


        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            # landmark_list=top_image["faces"],
            # connections=mp_face_mesh.FACE_CONNECTIONS,
            connections=face_components.components["lips"],
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)
    cv2.imshow('MediaPipe FaceMesh', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
