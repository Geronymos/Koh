import cv2

# class KeyEvents():

def key_events():
  key_action = {
    "a": "ever",
    "f": "face",
    "o": "face_oval",
    "m": "lips",
    "r": "right_eye",
    "y": "left_eyebrow",
    "l": "left_eye",
    "x": "right_eyebrow",
  }

  key = cv2.waitKey(1) & 0xFF

  # if key 
  # self.feature_switcher.switch_state()
  global running

  if key is ord("q"): 
    running = False
  elif key is ord("a"): # all
    feature_switcher.switch_state("every")
  elif key is ord("f"): # face
    feature_switcher.switch_state("face")
  elif key is ord("o"): # oval
    feature_switcher.switch_state("face_oval")
  elif key is ord("m"): # mouth/lips
    feature_switcher.switch_state("lips")
  elif key is ord("r"): # right eye
    feature_switcher.switch_state("right_eye")
  elif key is ord("y"): # left eyebrow
    feature_switcher.switch_state("left_eyebrow")
  elif key is ord("l"): # left eye
    feature_switcher.switch_state("left_eye")
  elif key is ord("x"): # right eyebrow
    feature_switcher.switch_state("right_eyebrow")

  if key > -1:
    features = feature_switcher.get_combined()["show"]
    for face in dataset:
      face["important"] = []
      for important in features:
        face["important"].extend(face["triplets"][important[0]])
