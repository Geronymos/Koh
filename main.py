import cv2
import mediapipe as mp
import json
import math

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


with open("faces.json", "r") as file:
  images_faces = json.load(file)

def match_faces(face1, face2):
  # dist1 = [[landmark["x"], landmark["y"], landmark["z"]] for landmark in face1]
  dist1 = [[landmark["x"], landmark["y"]] for landmark in face1]
  dist1 = [item for sublist in dist1 for item in sublist]
  # print(len(dist1))

  # dist2 = [[landmark["x"], landmark["y"], landmark["z"]] for landmark in face2]
  dist2 = [[landmark["x"], landmark["y"]] for landmark in face2]
  dist2 = [item for sublist in dist2 for item in sublist]
  # print(len(dist2))

  return math.dist(dist1, dist2)

# For static images:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

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
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:

        face_dict = [{"x": landmark.x, "y": landmark.y, "z": landmark.z} for landmark in face_landmarks.landmark]

        # print(face_dict)

        sorted_images = sorted(images_faces, key=lambda elem: match_faces(elem["faces"][0], face_dict))
        top_image = sorted_images[0]["file"]
        print(top_image)

        image_match = cv2.imread(top_image)
        cv2.imshow('Selected image', image_match)

        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)
    cv2.imshow('MediaPipe FaceMesh', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
