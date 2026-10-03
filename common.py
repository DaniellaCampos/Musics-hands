"""Piezas compartidas entre extracción, entrenamiento e inferencia."""
import numpy as np
import torch
from torch import nn

# (carpeta del dataset, etiqueta para mostrar, valor numérico o None si no es un número)
CLASSES = [
    ("acorde_do_1dedo", "1 dedo (Do)", 1),
    ("acorde_re_2dedos", "2 dedos (Re)", 2),
    ("acorde_mi_3dedos", "3 dedos (Mi)", 3),
    ("acorde_fa_4dedos", "4 dedos (Fa)", 4),
    ("acorde_sol_5dedos", "5 dedos (Sol)", 5),
    ("acorde_la_rock", "Rock (La)", None),
    ("acorde_si_pinch", "Pinch (Si)", None),
]
CLASS_FOLDERS = [c[0] for c in CLASSES]
NUM_FEATURES = 21 * 3


def normalize_landmarks(landmarks, width, height):
    """21 landmarks de MediaPipe -> vector (63,) invariante a posición y tamaño de la mano.

    `landmarks` es un objeto `NormalizedLandmarkList` o un array (21, 3) con coords en [0, 1].
    Se pasan a píxeles para no deformar por el aspect ratio, se centran en la muñeca y se
    escalan por la distancia máxima a la muñeca.
    """
    if hasattr(landmarks, "landmark"):
        landmarks = [(p.x, p.y, p.z) for p in landmarks.landmark]
    pts = np.asarray(landmarks, dtype=np.float32).copy()
    pts[:, 0] *= width
    pts[:, 1] *= height
    pts[:, 2] *= width  # MediaPipe usa aprox. la misma escala que x para z
    pts -= pts[0]
    scale = np.linalg.norm(pts[:, :2], axis=1).max()
    if scale > 0:
        pts /= scale
    return pts.reshape(-1)


class HandMLP(nn.Module):
    def __init__(self, num_classes=len(CLASSES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_FEATURES, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def augment(x):
    """Augmentación sobre un batch (B, 63): espejo, rotación 2D, escala y ruido."""
    b = x.shape[0]
    pts = x.view(b, 21, 3).clone()
    # Espejo horizontal: el modelo sirve igual para mano izquierda y derecha
    flip = torch.rand(b) < 0.5
    pts[flip, :, 0] *= -1
    # Rotación ±15° en el plano de la imagen
    ang = (torch.rand(b) * 2 - 1) * (15 * np.pi / 180)
    cos, sin = torch.cos(ang)[:, None], torch.sin(ang)[:, None]
    px, py = pts[:, :, 0].clone(), pts[:, :, 1].clone()
    pts[:, :, 0] = cos * px - sin * py
    pts[:, :, 1] = sin * px + cos * py
    # Escala ±10% y ruido gaussiano
    pts *= (1 + (torch.rand(b) * 2 - 1) * 0.1)[:, None, None]
    pts += torch.randn_like(pts) * 0.01
    return pts.view(b, -1)


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = HandMLP(len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["classes"]
