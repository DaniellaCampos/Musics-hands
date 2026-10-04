// Port de common.normalize_landmarks y count_fingers.count_fingers (mismos umbrales que Python).

const FINGERS = [[5, 6, 8], [9, 10, 12], [13, 14, 16], [17, 18, 20]]; // MCP, PIP, TIP
const FINGER_ANGLE = 140;
const THUMB_DIST = 0.6;
const PINCH_DIST = 0.3;

/** 21 landmarks {x,y,z} en [0,1] -> array (21) de [x,y,z] centrado en la muñeca y escalado. */
export function normalizeLandmarks(landmarks, width, height) {
  const [wx, wy, wz] = [landmarks[0].x * width, landmarks[0].y * height, landmarks[0].z * width];
  const pts = landmarks.map((p) => [p.x * width - wx, p.y * height - wy, p.z * width - wz]);
  const scale = Math.max(...pts.map((p) => Math.hypot(p[0], p[1])));
  if (scale > 0) for (const p of pts) { p[0] /= scale; p[1] /= scale; p[2] /= scale; }
  return pts;
}

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

function angle(a, b, c) {
  const v1 = [a[0] - b[0], a[1] - b[1]];
  const v2 = [c[0] - b[0], c[1] - b[1]];
  const cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (Math.hypot(...v1) * Math.hypot(...v2) + 1e-9);
  return (Math.acos(Math.min(1, Math.max(-1, cos))) * 180) / Math.PI;
}

/** Devuelve { count, fingers: [pulgar, índice, medio, anular, meñique] }. */
export function countFingers(pts) {
  const palm = dist(pts[0], pts[9]);
  const up = FINGERS.map(([mcp, pip, tip]) =>
    angle(pts[mcp], pts[pip], pts[tip]) > FINGER_ANGLE && dist(pts[tip], pts[0]) > dist(pts[pip], pts[0]));
  let thumb = dist(pts[4], pts[9]) / palm > THUMB_DIST;
  const pinch = dist(pts[4], pts[8]) / palm < PINCH_DIST;
  if (pinch) { thumb = false; up[0] = false; }
  const fingers = [thumb, ...up];
  return { count: fingers.filter(Boolean).length, fingers, pinch };
}

/**
 * Gesto del dataset solo con reglas sobre los landmarks de MediaPipe.
 * Devuelve el índice de clase (0=1 dedo … 4=5 dedos, 5=rock, 6=pinch) o null (puño).
 */
export function classifyGesture({ count, fingers, pinch }) {
  const [thumb, index, middle, ring, pinky] = fingers;
  if (pinch && middle && ring && pinky) return 6;
  if (index && pinky && !middle && !ring) return thumb ? 2 : 5; // te quiero (3) / rock
  return count >= 1 ? count - 1 : null;
}
