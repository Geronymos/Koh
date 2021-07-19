#!/usr/bin/env python

import cv2
import mediapipe as mp
from os import listdir
from os.path import isfile, join
import sys
import json
import csv
from tqdm import tqdm

mypath = sys.argv[1]
save_to=mypath.replace("/", "_")

onlyfiles = [f"{mypath}/{f}" for f in listdir(mypath) if isfile(join(mypath, f))]

mp_face_mesh = mp.solutions.face_mesh

# For static images:
IMAGE_FILES = onlyfiles

images_faces = []

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5) as face_mesh:
  pbar = tqdm(IMAGE_FILES)

  with open(save_to + ".csv", "w") as output_file:
    faces_writer = csv.writer(output_file)

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

      faces = [[{"x": landmark.x, "y": landmark.y, "z": landmark.z} for landmark in face.landmark] for face in results.multi_face_landmarks]

      images_faces.append({
        "index": idx,
        "file": file,
        "faces": faces
      })

      xyz_list = []
      for face in results.multi_face_landmarks:
        for landmark in face.landmark:
          xyz_list.append(landmark.x)
          xyz_list.append(landmark.y)
          xyz_list.append(landmark.z)

      faces_writer.writerow([file, *xyz_list])

      pbar.write(file)
      # pbar.set_postfix_str(file)
      # pbar.set_description(file)
      pbar.update(1)

  with open(save_to + ".json", "w") as file:
    json.dump(images_faces, file)