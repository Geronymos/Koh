import cv2
from components.FaceConverter import FaceConverter

class TrackingHelper:
  def __init__(self, model, image):
    self.image = image
    self.height, self.width, _color = self.image.shape
    self.faces = []
    
    # flip
    self.image = cv2.cvtColor(cv2.flip(self.image, 1), cv2.COLOR_BGR2RGB)
    self.image.flags.writeable = False

    # track face
    results = model.process(self.image)

    self.image.flags.writeable = True
    self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

    for face in results.multi_face_landmarks or []:
      face_converter = FaceConverter()
      face_converter.from_face(face)
      self.faces.append(face_converter)