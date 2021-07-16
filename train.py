import cv2
import mediapipe as mp
from os import listdir
from os.path import isfile, join
import json
from tqdm import tqdm

mypath = "dataset/vgg_face_dataset/images"
# mypath = "dataset/thispersondoesnotexist"
save_to=mypath.replace("/", "_") + ".json"

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
  pbar = tqdm(IMAGE_FILES)
  for idx, file in enumerate(pbar):
    image = cv2.imread(file)
    # Convert the BGR image to RGB before processing.
    try:
      results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    except:
      continue

    # Print and draw face mesh landmarks on the image.
    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()

    faces = [[{"x": landmark.x, "y": landmark.y, "z": landmark.z} for landmark in face.landmark] for face in results.multi_face_landmarks]
    # print(faces)
    
    # print(multi_face_landmarks_dict)

    images_faces.append({
      "index": idx,
      "file": file,
      "faces": faces
    })

    pbar.write(file)
    # pbar.set_postfix_str(file)
    # pbar.set_description(file)
    pbar.update(1)

  with open(save_to, "w") as file:
    json.dump(images_faces, file)