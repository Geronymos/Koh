from components.FeatureSwitcher import FeatureSwitcher
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark 
import csv

feature_switcher = FeatureSwitcher()

class Face:
  def __init__(self, landmarks = []):
    self.landmark = landmarks

def read_dataset(path):
  with open(path, "r") as file:
    csv_reader = csv.reader(file)

    dataset = []

    for image_file, *positions in csv_reader:
      
      positions = [float(position) for position in positions]

      landmarks = []

      triplets = [*zip(*(iter(positions),) * 3)]

      for x,y,z in triplets:
        normalized_landmark = NormalizedLandmark()
        normalized_landmark.x = x
        normalized_landmark.y = y
        normalized_landmark.z = z
        landmarks.append(normalized_landmark)

      features = feature_switcher.get_combined()["show"]
      important = []
      for feature in features:
        important.extend(triplets[feature[0]])

      dataset.append({
        "filename": image_file,
        "landmarks": Face(landmarks),
        "vertices": positions,
        "triplets": triplets,
        "important": important
      })

    return dataset
