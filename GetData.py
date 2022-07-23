import cv2
import mediapipe as mp
import csv
import numpy as np
import os


def get_lms(img: np.array):
    h, w, _ = img.shape
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
class_name = "Surprises"
filename = "SurprisedPoints.csv"
start = False
while True:
    success, img = cap.read()
    if success is False:
        break

    # img = cv2.flip(img, 1)
    # img = cv2.resize(img, (480, 640))
    img = cv2.resize(img, (640, 480))

    lms_list, num_row = get_lms(img)
    if lms_list:
        cv2.rectangle(img, (150, 100), (450, 400), (255, 0, 255), 3)

        if os.path.exists(filename) is False:
            columns = ['class']
            for val in range(1, num_row + 1):
                columns += [f"x{val}", f"y{val}", f"z{val}", f"v{val}"]

            with open(filename, mode="w", newline="") as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(columns)

        if start:
            print("getting data...")
            lms_list.insert(0, class_name)
            with open(filename, mode="a", newline="") as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(lms_list)

    cv2.imshow("Res", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

    if key == ord('s'):
        start = True

cap.release()
cv2.destroyAllWindows()
