import mediapipe as mp

class LandmarkConverter:
  def __init__(self):
    self.x = 0
    self.y = 0
    self.z = 0
  def from_landmark(self, landmark):
    self.x = landmark.x
    self.y = landmark.y
    self.z = landmark.z
  def from_vertex(self, vertex):
    x,y,z = vertex
    self.x = x
    self.y = y
    self.z = z
  def to_landmark(self):
    normalized_landmark = mp.framework.formats.landmark_pb2.NormalizedLandmark()
    normalized_landmark.x = self.x
    normalized_landmark.y = self.y
    normalized_landmark.z = self.z
    return normalized_landmark
  def to_vertex(self):
    return [self.x, self.y, self.z]