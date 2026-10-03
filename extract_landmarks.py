"""Paso 1: extrae los 21 landmarks de cada imagen del dataset con MediaPipe -> landmarks.npz"""
import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from common import CLASS_FOLDERS, normalize_landmarks


def enhance_contrast(img):
    """CLAHE sobre la luminancia: rescata manos a contraluz (p. ej. con un foco detrás)."""
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def detect(hands, img):
    for candidate in (img, enhance_contrast(img)):
        res = hands.process(cv2.cvtColor(candidate, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            return res.multi_hand_landmarks[0]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset_unificado")
    parser.add_argument("--out", default="landmarks.npz")
    parser.add_argument("--min-detection-confidence", type=float, default=0.1)
    args = parser.parse_args()

    X, y, failed = [], [], []
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
                                  min_detection_confidence=args.min_detection_confidence) as hands:
        for label, folder in enumerate(CLASS_FOLDERS):
            files = sorted((Path(args.dataset) / folder).glob("*.jpg"))
            ok = 0
            for f in files:
                # imdecode en vez de imread para soportar rutas con caracteres no ASCII
                img = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
                landmarks = detect(hands, img)
                if landmarks is None:
                    failed.append(str(f))
                    continue
                h, w = img.shape[:2]
                X.append(normalize_landmarks(landmarks, w, h))
                y.append(label)
                ok += 1
            print(f"{folder:20s} {ok:4d}/{len(files):4d} detectadas ({100 * ok / max(len(files), 1):.1f}%)")

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    np.savez_compressed(args.out, X=X, y=y, classes=np.array(CLASS_FOLDERS))
    print(f"\nGuardado {args.out}: X={X.shape}, y={y.shape}, sin mano detectada={len(failed)}")
    if failed:
        Path("sin_deteccion.txt").write_text("\n".join(failed), encoding="utf-8")
        print("Lista de imágenes sin detección en sin_deteccion.txt")


if __name__ == "__main__":
    main()
