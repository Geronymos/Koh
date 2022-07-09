import mediapipe as mp
import cv2

mp_face_mesh = mp.solutions.face_mesh

class Tracking():
    def __init__(self):

        #with mp_face_mesh.FaceMesh( max_num_faces=1, min_detection_confidence=0.5 ) as face_mesh:
        #    self.face_mesh = face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh( max_num_faces=1, min_detection_confidence=0.5 )

    def track(self, webcam_image):

      width, height, _color = webcam_image.shape
      # Flip the image horizontally for a later selfie-view display, and convert
      # the BGR image to RGB.
      webcam_image = cv2.cvtColor(cv2.flip(webcam_image, 1), cv2.COLOR_BGR2RGB)
      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      webcam_image.flags.writeable = False
      results = self.face_mesh.process(webcam_image)

      # Draw the face mesh annotations on the image.
      webcam_image.flags.writeable = True
      webcam_image = cv2.cvtColor(webcam_image, cv2.COLOR_RGB2BGR)

      # features = feature_switcher.get_combined()
      
      return results

