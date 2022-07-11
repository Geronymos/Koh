import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec_webcam = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
drawing_spec_match = mp_drawing.DrawingSpec(thickness=1, circle_radius=1,color=(0,0,255))

class Drawing():
    def __init__(self):
        cv2.namedWindow('Selected image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Selected image', 800, 600)

    def drawWebcam(self, webcam):
        cv2.imshow('MediaPipe FaceMesh', webcam)

    def drawStates(states, webcam):
        for i, [key, value] in enumerate(feature_switcher.states.items()):
            cv2.putText(
                    webcam_image,
                    f"{key}: {value}",
                    (width - 100, 10 * (i+1)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    .3, 
                    (255, 255, 255, 255),
                    1
                    )

    def drawMatch(self, match, webcam):

        image_match = cv2.imread(match["image"])

        cv2.imshow('Selected image', image_match)
        mp_drawing.draw_landmarks(
              image=webcam,
              landmark_list=match["face"],
              # connections=feature_switcher.get_combined()["match"],
              connections=mp_face_mesh.FACEMESH_TESSELATION,
              landmark_drawing_spec=drawing_spec_match,
              connection_drawing_spec=drawing_spec_match)


    def drawFace(self, results, webcam):
      # show webcam image with wireframe
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                      image=webcam,
                      landmark_list=face_landmarks,
                      # connections=features["match"],
                      connections=mp_face_mesh.FACEMESH_TESSELATION,
                      landmark_drawing_spec=drawing_spec_webcam,
                      connection_drawing_spec=drawing_spec_webcam)
