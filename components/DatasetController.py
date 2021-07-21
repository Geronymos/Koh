import math


class DatasetController:
  def __init__(self, face_converter_list):
    self.faces = face_converter_list
    

  def match(self, cam_face, features):
    # faces_vertices_list = [face.get_important_features(features).to_vertices_list() for face in self.faces]
    cam_face_vertices = cam_face.to_vertices_list()

    matches = sorted(self.faces, key=lambda elem: math.dist(elem.get_important_features(features).to_vertices_list() , cam_face_vertices))

    return matches
