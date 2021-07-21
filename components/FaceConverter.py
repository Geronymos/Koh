from components.LandmarkConverter import LandmarkConverter
import mediapipe as mp
from itertools import chain

class FaceConverter:
  def __init__(self, image_name = "", landmark_converters = []):
    self.image = image_name
    self.landmarks = landmark_converters

  def from_face(self, face):
    for landmark in face.landmark:
      landmark_converter = LandmarkConverter()
      landmark_converter.from_landmark(landmark)
      self.landmarks.append(landmark_converter)
  
  def from_vertices_list(self, vertices_list):
    for vertex in zip(*(iter(vertices_list),) * 3):
      landmark_converter = LandmarkConverter()
      landmark_converter.from_vertex(vertex)
      self.landmarks.append(landmark_converter)
  
  def to_face(self):
    normalized_landmark_list = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
    for landmark in self.landmarks:
      normalized_landmark_list.landmark.append(landmark.to_landmark())
    return normalized_landmark_list
  
  def to_vertices_list(self):
    vertices_list = []
    for landmark in self.landmarks:
      vertices_list.extend(landmark.to_vertex())
    
    return vertices_list

  def get_important_features(self, features):
    important_indices = set(chain.from_iterable(features))
    important_landmarks = FaceConverter([self.landmarks[i] for i in important_indices])

    return important_landmarks





