
import csv
import mediapipe as mp
from components.FeatureSwitcher   import FeatureSwitcher 

class Face:
  def __init__(self, landmarks = []):
    self.landmark = landmarks

def load(dataset_filename):
    with open(dataset_filename, "r") as file:
      csv_reader = csv.reader(file)

      feature_switcher = FeatureSwitcher()

      dataset = []

      for image_file, *positions in csv_reader:
        
        positions = [float(position) for position in positions]

        landmarks = []

        triplets = [*zip(*(iter(positions),) * 3)]

        for x,y,z in triplets:
          normalized_landmark = mp.framework.formats.landmark_pb2.NormalizedLandmark()
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
