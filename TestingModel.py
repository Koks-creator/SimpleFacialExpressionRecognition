import cv2
import mediapipe as mp
import numpy as np
import pickle
import pandas as pd


def get_lms(img: np.array):
    h, w, _ = img.shape
    # img.flags.writeable = False
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results_raw = face.process(img_rgb)
    results = results_raw.multi_face_landmarks
    face_mesh_list = []

    if results:
        for landmark in results:
            mp_draw.draw_landmarks(img, landmark, mp_face_mesh.FACE_CONNECTIONS)

            for lm in landmark.landmark:
                face_mesh_list.append([lm.x, lm.y, lm.z, lm.visibility])

    return list(np.array(face_mesh_list).flatten()), len(face_mesh_list)


mp_face_mesh = mp.solutions.face_mesh
face = mp_face_mesh.FaceMesh()
mp_draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)

with open(r"FacialExpressionModel.pkl", "rb") as f:
    model = pickle.load(f)


while True:
    success, img = cap.read()
    if success is False:
        break

    img = cv2.resize(img, (640, 480))

    lms_list, num_row = get_lms(img)
    if lms_list:
        cv2.rectangle(img, (150, 100), (450, 400), (255, 0, 255), 3)
        x = pd.DataFrame([lms_list])
        body_lang_class = model.predict(x)[0]
        body_lang_prob = model.predict_proba(x)[0]
        prob = body_lang_prob[np.argmax(body_lang_prob)]
        cv2.putText(img, f"{body_lang_class} {prob}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    cv2.imshow("Res", img)
    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()
