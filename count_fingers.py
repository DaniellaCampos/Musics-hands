"""Cuenta dedos levantados usando solo los landmarks de MediaPipe (reglas geométricas, sin entrenar).

Webcam:            python count_fingers.py
Medir en dataset:  python count_fingers.py --eval      (usa landmarks.npz)
"""
import argparse

import numpy as np

from common import CLASS_FOLDERS, normalize_landmarks

# (MCP, PIP, TIP) de índice, medio, anular y meñique
FINGERS = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]
FINGER_ANGLE = 140   # grados en el PIP: más recto que esto = dedo extendido
THUMB_DIST = 0.6     # punta del pulgar lejos de la base del dedo medio (en tamaños de palma)
PINCH_DIST = 0.3     # punta del pulgar tocando la del índice = pinch, ninguno de los dos cuenta


def _angle(a, b, c):
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def count_fingers(features):
    """`features`: vector (63,) de normalize_landmarks. Devuelve (n_dedos, lista de bools por dedo)."""
    p = np.asarray(features).reshape(21, 3)[:, :2]
    dist = lambda i, j: np.linalg.norm(p[i] - p[j])
    palm = dist(0, 9)

    up = []
    for mcp, pip, tip in FINGERS:
        straight = _angle(p[mcp], p[pip], p[tip]) > FINGER_ANGLE
        away = dist(tip, 0) > dist(pip, 0)
        up.append(bool(straight and away))
    thumb = dist(4, 9) / palm > THUMB_DIST

    if dist(4, 8) / palm < PINCH_DIST:
        thumb, up[0] = False, False
    fingers = [thumb] + up  # pulgar, índice, medio, anular, meñique
    return sum(fingers), fingers


def evaluate(path):
    # Dedos levantados esperados por clase del dataset
    expected = {"acorde_do_1dedo": 1, "acorde_re_2dedos": 2, "acorde_mi_3dedos": 3, "acorde_fa_4dedos": 4,
                "acorde_sol_5dedos": 5, "acorde_la_rock": 2, "acorde_si_pinch": 3}
    data = np.load(path)
    hits = np.array([count_fingers(x)[0] == expected[CLASS_FOLDERS[c]] for x, c in zip(data["X"], data["y"])])
    for i, folder in enumerate(CLASS_FOLDERS):
        print(f"{folder:20s} {hits[data['y'] == i].mean():.3f}")
    print(f"{'TOTAL':20s} {hits.mean():.3f}")


def webcam(camera):
    import cv2
    import mediapipe as mp

    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(camera)
    with mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.6,
                                  min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            total = 0
            if res.multi_hand_landmarks:
                for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                    draw.draw_landmarks(frame, lm, mp.solutions.hands.HAND_CONNECTIONS)
                    n, _ = count_fingers(normalize_landmarks(lm, w, h))
                    total += n
                    side = "Izq" if handed.classification[0].label == "Left" else "Der"
                    wx, wy = int(lm.landmark[0].x * w), int(lm.landmark[0].y * h)
                    cv2.putText(frame, f"{side}: {n}", (max(wx - 40, 0), min(wy + 30, h - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(frame, f"Total: {total}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)

            cv2.imshow("Contador de dedos", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="medir precisión sobre landmarks.npz")
    parser.add_argument("--data", default="landmarks.npz")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    evaluate(args.data) if args.eval else webcam(args.camera)
