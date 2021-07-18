import cv2
import mediapipe as mp
import json
import math

import face_components

# landmark_pb2.NormalizedLandmarkList

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

trained_file = "dataset_thispersondoesnotexist.json"
trained_file = "dataset_friends.json"
trained_file = "dataset_partyAllTheTime.json"
trained_file = "dataset_rickroll.json"
trained_file = "dataset_jim_carrey.json"
trained_file = "dataset_vgg_face_dataset_images.json"
trained_file = "dataset_kaggle.json"

connectionsSwitcher = face_components.ConnectionsSwitcher()

def xyz_list_from_face(face):
  # important_landmarks = [face[important[0]] for important in mp_face_mesh.FACE_CONNECTIONS]
  important_landmarks = [face[important[0]] for important in connectionsSwitcher.get_combined()["show"]]
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

class Koh:
  def __init__(self, images_faces):
    self.data = images_faces

    self.drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    self.video_capture = cv2.VideoCapture(0)

    self.running = True

    cv2.namedWindow('Selected image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Selected image', 800, 600)
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        # min_tracking_confidence=0.5
        ) as self.face_mesh:
      while self.video_capture.isOpened() and self.running:
        success, image = self.video_capture.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue

        self.render(image)
        self.key_events()
    
    self.video_capture.release()

  def render(self, image):
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False


    results = self.face_mesh.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    height, width, color = image.shape


    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:

        face_dict = [{"x": landmark.x, "y": landmark.y, "z": landmark.z} for landmark in face_landmarks.landmark]

        face_xyz = xyz_list_from_face(face_dict)

        sorted_images = sorted(images_faces, key=lambda elem: match_faces(elem["xyz"], face_xyz))
        top_image = sorted_images[0]["file"]
        print(top_image)

        image_match = cv2.imread(top_image)

        cv2.imshow('Selected image', image_match)

        for i, key in enumerate(connectionsSwitcher.states):
          cv2.putText(
            image,
            f"{key}: {connectionsSwitcher.states[key]}",
            (width - 100, 10 * (i+1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            .3, 
            (255, 255, 255, 255),
            1
          )

        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=connectionsSwitcher.get_combined()["match"],
            landmark_drawing_spec=self.drawing_spec,
            connection_drawing_spec=self.drawing_spec)

    cv2.imshow('MediaPipe FaceMesh', image)

  def key_events(self):
    key = cv2.waitKey(1) & 0xFF
    if key is ord("q"): 
      self.running = False
    elif key is ord("a"): # all
      connectionsSwitcher.switch_state("every")
    elif key is ord("f"): # face
      connectionsSwitcher.switch_state("face")
    elif key is ord("o"): # oval
      connectionsSwitcher.switch_state("face_oval")
    elif key is ord("m"): # mouth/lips
      connectionsSwitcher.switch_state("lips")
    elif key is ord("r"): # right eye
      connectionsSwitcher.switch_state("right_eye")
    elif key is ord("y"): # left eyebrow
      connectionsSwitcher.switch_state("left_eyebrow")
    elif key is ord("l"): # left eye
      connectionsSwitcher.switch_state("left_eye")
    elif key is ord("x"): # right eyebrow
      connectionsSwitcher.switch_state("right_eyebrow")

many_faces = Koh(images_faces)