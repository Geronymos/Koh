
from scipy.spatial import KDTree
import csv
import mediapipe as mp
from components.FeatureSwitcher   import FeatureSwitcher 

class Face:
    def __init__(self, landmarks = []):
        self.landmark = landmarks

def load(dataset_filename):
    with open(dataset_filename, "r") as file:
        csv_reader = csv.reader(file)
        
        # feature_switcher = FeatureSwitcher()

        dataset = []
        points = []
        files = []
        faces = []
        for f, *p in csv_reader:
            files.append(f)
            positions = [float(e) for e in p]
            points.append(positions)
            landmarks = []

            triplets = [*zip(*(iter(positions),) * 3)]

            for x,y,z in triplets:
                normalized_landmark = mp.framework.formats.landmark_pb2.NormalizedLandmark()
                normalized_landmark.x = x
                normalized_landmark.y = y
                normalized_landmark.z = z
                landmarks.append(normalized_landmark)
             
            faces.append(Face(landmarks))


        return {
                "files": files,
                "tree": KDTree(points),
                "face": faces
                }

