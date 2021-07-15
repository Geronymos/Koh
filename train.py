import cv2
import mediapipe as mp
from os import listdir
from os.path import isfile, join
import json

mypath = "dataset"

onlyfiles = [f"{mypath}/{f}" for f in listdir(mypath) if isfile(join(mypath, f))]

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# print(onlyfiles)s

# For static images:
IMAGE_FILES = onlyfiles
# IMAGE_FILES = ["dataset/image_0.png"]
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

images_faces = []

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5) as face_mesh:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print and draw face mesh landmarks on the image.
    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()

    faces = [[{"x": landmark.x, "y": landmark.y, "z": landmark.z} for landmark in face.landmark] for face in results.multi_face_landmarks]
    print(faces)
    # print(multi_face_landmarks_dict)

    images_faces.append({
      "index": idx,
      "file": file,
      "faces": faces
    })

  with open("faces.json", "w") as file:
    json.dump(images_faces, file)