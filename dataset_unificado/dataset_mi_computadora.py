import os
import cv2

# 1. Las primeras 3 notas asignadas a tu compañera
gestos = ["acorde_do_1dedo", "acorde_re_2dedos", "acorde_mi_3dedos"]
dataset_dir = "dataset_parte_1"

# Crear carpetas automáticamente si no existen
for gesto in gestos:
  os.makedirs(os.path.join(dataset_dir, gesto), exist_ok=True)

# 2. Inicializar la cámara web
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

gesto_activo_idx = 0  # Inicia en el primer gesto (Do)
meta_fotos = 150

# Contar cuántas fotos ya existen (por si retoma el programa)
contador_fotos = {}
for gesto in gestos:
  ruta_carpeta = os.path.join(dataset_dir, gesto)
  archivos = os.listdir(ruta_carpeta)
  contador_fotos[gesto] = len(archivos)

print("==================================================")
print("COMPAÑERA - CAPTURA DE LAS 3 PRIMERAS NOTAS (Versión Limpia)")
print("Teclas '0', '1', '2': Cambiar de nota musical")
print("Barra ESPACIADORA: Tomar foto")
print("Tecla 'Q': Salir del programa")
print("==================================================")

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  frame = cv2.flip(frame, 1)

  # 👉 COPIA LIMPIA: Guardamos el frame sin textos ni barras para que el dataset quede limpio
  frame_limpio = frame.copy()

  h, w, _ = frame.shape
  gesto_actual = gestos[gesto_activo_idx]
  actual_count = contador_fotos[gesto_actual]

  # Panel visual superior (solo se ve en vivo en pantalla)
  overlay = frame.copy()
  cv2.rectangle(overlay, (10, 10), (w - 10, 165), (30, 30, 30), -1)
  cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

  cv2.putText(
      frame,
      f"PARTE 1 - GESTO: [{gesto_activo_idx}] {gesto_actual.upper()}",
      (20, 40),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.55,
      (0, 255, 255),
      2,
  )
  color_progreso = (0, 255, 0) if actual_count >= meta_fotos else (0, 165, 255)
  cv2.putText(
      frame,
      f"Fotos tomadas: {actual_count} / {meta_fotos} meta",
      (20, 75),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.6,
      color_progreso,
      2,
  )

  # Barra de estado visual
  bar_x, bar_y, bar_w, bar_h = 20, 95, w - 40, 20
  cv2.rectangle(
      frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1
  )
  porcentaje = min(actual_count / meta_fotos, 1.0)
  fill_w = int(bar_w * porcentaje)
  color_barra = (0, 255, 0) if actual_count >= meta_fotos else (0, 140, 255)
  if fill_w > 0:
    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + fill_w, bar_y + bar_h),
        color_barra,
        -1,
    )
  cv2.rectangle(
      frame,
      (bar_x, bar_y),
      (bar_x + bar_w, bar_y + bar_h),
      (255, 255, 255),
      1,
  )

  if actual_count >= meta_fotos:
    cv2.putText(
        frame,
        "¡META CUMPLIDA! Cambia con teclas 0, 1, 2",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )
  else:
    cv2.putText(
        frame,
        "Presiona ESPACIO para capturar variaciones",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
    )

  cv2.putText(
      frame,
      "ESPACIO: Foto  |  TECLAS 0-2: Cambiar Nota  |  Q: Salir",
      (20, h - 20),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.45,
      (255, 255, 255),
      1,
  )

  cv2.imshow("Dataset - Compañera (Parte 1)", frame)

  key = cv2.waitKey(1) & 0xFF
  if key == ord("q"):
    break
  elif ord("0") <= key <= ord("2"):  # Solo teclas 0, 1 y 2 para sus 3 notas
    gesto_activo_idx = key - ord("0")
    print(f">> Cambiaste al acorde: {gestos[gesto_activo_idx]}")
  elif key == ord(" "):
    contador_fotos[gesto_actual] += 1
    nombre_archivo = os.path.join(
        dataset_dir,
        gesto_actual,
        f"{gesto_actual}_{contador_fotos[gesto_actual]}.jpg",
    )
    # 👉 GUARDAMOS LA FOTO LIMPIA SIN TEXTOS
    cv2.imwrite(nombre_archivo, frame_limpio)
    print(
        f"[GUARDADA LIMPIA] {nombre_archivo} -> Total en esta nota:"
        f" {contador_fotos[gesto_actual]}"
    )

cap.release()
cv2.destroyAllWindows()
print("¡Captura de la Parte 1 finalizada!")