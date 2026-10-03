"""Paso 3: reconocimiento en tiempo real de hasta 2 manos con la webcam. Tecla q para salir."""
import argparse

import cv2
import mediapipe as mp
import torch

from common import CLASSES, load_model, normalize_landmarks

LABELS = {folder: (label, value) for folder, label, value in CLASSES}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hand_model.pt")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()

    model, classes = load_model(args.model)
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(args.camera)

    with mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.6,
                                  min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            # Vista espejo: así MediaPipe etiqueta Left/Right correctamente desde el punto de vista del usuario
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            values = []
            if res.multi_hand_landmarks:
                for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                    draw.draw_landmarks(frame, lm, mp.solutions.hands.HAND_CONNECTIONS)
                    x = torch.from_numpy(normalize_landmarks(lm, w, h))[None]
                    with torch.no_grad():
                        probs = torch.softmax(model(x), 1)[0]
                    conf, idx = probs.max(0)
                    side = "Izq" if handed.classification[0].label == "Left" else "Der"
                    if conf >= args.threshold:
                        label, value = LABELS[classes[idx]]
                        values.append(value)
                    else:
                        label, value = "?", None
                        values.append(None)
                    wx = int(lm.landmark[0].x * w)
                    wy = int(lm.landmark[0].y * h)
                    cv2.putText(frame, f"{side}: {label} ({conf:.2f})", (max(wx - 80, 0), min(wy + 30, h - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if len(values) == 2 and all(v is not None for v in values):
                cv2.putText(frame, f"Total: {sum(values)}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)

            cv2.imshow("Hand numbers", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
