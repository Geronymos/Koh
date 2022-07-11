def match(face_results, dataset):
    if not face_results.multi_face_landmarks:
        return {
                "image": dataset["files"][0],
                "face": dataset["face"][0]
                }

    for face_landmarks in face_results.multi_face_landmarks:

        # important_webcam = [face_landmarks.landmark[important[0]] for important in features["show"]]
        important_webcam = face_landmarks.landmark
        # [*chain([face["triplets"][important[0]] for important in features])]

        face_vertices = []
        for landmark in important_webcam:
            face_vertices.append(landmark.x)
            face_vertices.append(landmark.y)
            face_vertices.append(landmark.z)

        # print(len(dataset[0]["important"]), len(face_vertices))

        index = dataset["tree"].query(face_vertices)[1]

        return {
                "image": dataset["files"][index],
                "face": dataset["face"][index]
                }
