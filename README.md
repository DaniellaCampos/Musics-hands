# Musics Hands

Reconocimiento de gestos de la mano para tocar acordes musicales. El proyecto tiene dos partes:

1. **Entrenamiento distribuido** de un clasificador de gestos (MLP) en **2 computadoras en paralelo**, con PyTorch DistributedDataParallel.
2. **Dashboard web** que, con la cámara, detecta hasta 2 manos y toca un acorde según el gesto de cada una.

---

## 1. Dataset

Dataset propio, hecho a mano: fotos de una mano haciendo un gesto, una mano por imagen, en `dataset_unificado/`.

| Carpeta | Gesto | Imágenes |
|---|---|---|
| `acorde_do_1dedo` | 1 dedo (índice) | 150 |
| `acorde_re_2dedos` | 2 dedos (índice y medio) | 151 |
| `acorde_mi_3dedos` | "Te quiero" (pulgar, índice y meñique) | 150 |
| `acorde_fa_4dedos` | 4 dedos (pulgar doblado) | 152 |
| `acorde_sol_5dedos` | 5 dedos | 150 |
| `acorde_la_rock` | Rock (índice y meñique) | 152 |
| `acorde_si_pinch` | Pinch / OK (pulgar e índice juntos) | 150 |
| **Total** | | **1055** |

- Todas las imágenes son JPG de 640×480.
- No tienen anotaciones de dónde está la mano; la etiqueta es la carpeta.
- Cada clase corresponde a un acorde de la escala de Do mayor (Do, Re, Mi, Fa, Sol, La, Si).

## 2. Extracción de características (`extract_landmarks.py`)

En vez de entrenar con píxeles, cada imagen se convierte en los **21 puntos clave de la mano** (landmarks x, y, z). Para eso se usa el detector de manos preentrenado de **MediaPipe Hands**.

- Se usa `static_image_mode=True` y `max_num_hands=1`.
- **Umbral de detección bajo (0.1).** Con el valor por defecto se perdía el 37% de la clase "1 dedo", porque muchas fotos tienen un foco detrás de la mano (contraluz).
- **Reintento con contraste mejorado.** Si no se detecta la mano, se repite con **CLAHE** sobre la luminancia, que recupera las manos a contraluz.

Resultado: **1050 de 1055 imágenes** con la mano detectada. Las 5 que fallan quedan listadas en `sin_deteccion.txt`.

| Clase | Detectadas |
|---|---|
| 1 dedo | 145 / 150 (96.7%) |
| 2 dedos | 151 / 151 (100%) |
| Te quiero | 150 / 150 (100%) |
| 4 dedos | 152 / 152 (100%) |
| 5 dedos | 150 / 150 (100%) |
| Rock | 152 / 152 (100%) |
| Pinch | 150 / 150 (100%) |

**Normalización** (`common.normalize_landmarks`), para que no importe dónde está la mano ni a qué distancia:

1. Las coordenadas se pasan a píxeles, para no deformar por la relación de aspecto 4:3.
2. Se centran en la muñeca (punto 0).
3. Se dividen por la distancia máxima a la muñeca.

El resultado es un vector de 63 valores por mano, guardado en `landmarks.npz`.

## 3. Modelo (`common.py`)

Perceptrón multicapa (MLP) pequeño:

```
63 → Linear(128) → ReLU → Dropout(0.2) → Linear(64) → ReLU → Dropout(0.2) → Linear(7)
```

- **Parámetros:** 16 903.
- **Pérdida:** entropía cruzada.
- **Optimizador:** Adam, lr = 1e-3, batch = 32, 80 épocas.

**Aumento de datos**, aplicado en cada batch (`common.augment`):

- Espejo horizontal aleatorio, para que funcione con la mano izquierda y la derecha.
- Rotación aleatoria de ±15°.
- Escala aleatoria de ±10%.
- Ruido gaussiano (σ = 0.01).

## 4. Entrenamiento distribuido en 2 computadoras (`train_ddp.py`)

### Cómo funciona

Se usa **data parallelism** con PyTorch `DistributedDataParallel` (DDP):

- **Mismo punto de partida:** las dos PCs tienen el mismo `landmarks.npz` y hacen la misma división 80/20 estratificada con semilla fija (840 muestras de entrenamiento y 210 de validación).
- **Datos repartidos:** `DistributedSampler` le da a cada PC **la mitad del entrenamiento**, 420 muestras.
- **Gradientes sincronizados:** en cada paso, cada PC calcula los gradientes con su mitad y DDP **los promedia entre las dos por la red**. Así los dos modelos quedan idénticos en todo momento.
- **Validación conjunta:** cada PC evalúa una parte de la validación y los aciertos se suman con `all_reduce`. Por eso ambas muestran la misma `val_acc(global)`.
- **Guardado:** al final, el rank 0 guarda `hand_model.pt` y muestra la matriz de confusión.

```
            PC 1 (rank 0, maestra)                 PC 2 (rank 1)
            420 muestras                           420 muestras
                  │ forward + backward                   │ forward + backward
                  └──────── all-reduce de gradientes ────┘
                            (gloo sobre Tailscale)
                  │ optimizer.step()                     │ optimizer.step()
                  ▼   pesos idénticos en ambas PCs        ▼
```

### Infraestructura

| Elemento | Elección | Motivo |
|---|---|---|
| Backend de comunicación | `gloo` | NCCL no existe en Windows |
| Dispositivo | CPU en ambas PCs | El modelo es muy pequeño; mezclar CPU y GPU complica gloo |
| Red | Tailscale (VPN) | Las PCs estaban en redes distintas; Tailscale les da IPs privadas `100.x.y.z` |
| Inicialización | `init_method="tcp://<IP_maestra>:29500"` | La maestra abre el punto de encuentro y la otra se conecta |

### Pasos que se siguieron

1. **Entorno en las dos PCs:** un entorno virtual con la misma versión de PyTorch (`2.14.1+cpu`).
2. **Red:** Tailscale con la misma cuenta en ambas PCs. La maestra quedó con la IP `100.113.242.111`.
3. **Firewall:** en las dos PCs se permitió `python.exe` del entorno virtual. Gloo abre puertos aleatorios además del 29500.
4. **Interfaz de red:** en cada terminal se ejecutó `$env:GLOO_SOCKET_IFNAME = "Tailscale"`, para que gloo use la VPN.
5. **Datos:** `landmarks.npz` se generó una vez en la maestra y se compartió por git.
6. **Entrenamiento:**
   ```powershell
   # PC maestra
   python train_ddp.py --rank 0 --world-size 2 --master-addr 100.113.242.111 --master-port 29500
   # PC 2
   python train_ddp.py --rank 1 --world-size 2 --master-addr 100.113.242.111 --master-port 29500
   ```

### Problemas encontrados y cómo se resolvieron

| Problema | Causa | Solución |
|---|---|---|
| `pip` no encontraba `mediapipe==0.10.21` en la PC 2 | La PC 2 tenía Python 3.14 y MediaPipe solo publica paquetes hasta 3.12 | La PC 2 solo entrena, así que basta con torch, numpy y scikit-learn |
| `No module named numpy` en la PC 2 | Como falló mediapipe, `pip install -r requirements.txt` canceló toda la instalación | Instalar las dependencias de entrenamiento por separado |
| `use_libuv was requested but PyTorch was built without libuv` | PyTorch para Windows no incluye libuv | `USE_LIBUV=0` dentro del script |
| `Connection was likely closed` en el rank 1 | Se interrumpió el rank 0 con Ctrl+C mientras esperaba | El rank 0 espera sin mostrar nada hasta que se une el rank 1; se agregó un mensaje que lo avisa |
| Aviso `client socket has failed to connect to [...]` | PyTorch prueba primero con el nombre del equipo | Es inofensivo; después conecta por IP |

### Resultados

**Ensayo previo** (2 procesos en una sola PC, 40 épocas):

- Cada proceso entrenó con 420 muestras y tardó 7.6 s.
- La exactitud en validación llegó a 1.000 en la época 25.
- La matriz de confusión no tuvo errores.

**Entrenamiento real entre las 2 PCs:**

| Métrica | Valor |
|---|---|
| Épocas | 80 |
| Tiempo total | _[completar con la salida de `train_ddp.py`]_ |
| `val_acc` final | _[completar]_ |

**Límite importante:** la exactitud en validación es optimista. Las fotos del dataset son ráfagas casi idénticas, de pocas personas y en pocos lugares, así que entrenamiento y validación comparten imágenes muy parecidas. **Con la cámara en vivo, el modelo generaliza mal** a otras manos, luces y ángulos, y a formas distintas de hacer el gesto (por ejemplo, "3" con índice, medio y anular, que no está en el dataset). Para mejorarlo hacen falta más datos y más variados.

## 5. Exportación a la web (`export_web_model.py`)

- **Exportación:** `export_web_model.py` guarda los pesos de `hand_model.pt` en `web/model.json`.
- **Ejecución en el navegador:** el archivo lo usa `web/js/model.js`, una implementación del MLP en JavaScript.
- **Verificación:** con las 1050 muestras del dataset, la versión JavaScript da **la misma predicción que PyTorch en 1050 de 1050**. La normalización y el conteo de dedos también se compararon contra Python, con 1050 de 1050 iguales.

## 6. Dashboard web (`web/`)

Corre todo en el navegador:

1. La cámara captura cada cuadro.
2. **MediaPipe Hands** (versión JavaScript, con GPU) detecta hasta 2 manos y sus 21 puntos. sirve como referencia visual pero el modelo es el que dice los gestos
3. **El gesto de cada mano**:
   - Un dedo está levantado si está recto en la articulación media y su punta queda más lejos de la muñeca que esa articulación.
   - El pulgar cuenta si su punta está lejos de la base del dedo medio.
   - Pinch: si las puntas del pulgar y el índice se tocan.
   - Con esas reglas se distinguen los 7 gestos; sobre el dataset aciertan el **99.1%**.
3. **Un sintetizador con Web Audio** toca el acorde de cada mano. La mano izquierda suena una octava más grave. Con el puño cerrado esa mano se calla.

**Modos:**
- **Acordes por gesto:** cada mano toca un acorde: 1 dedo → Do, 2 → Re m, te quiero → Mi m, 4 → Fa, 5 → Sol, rock → La m, pinch → Si°.
- **Escala por dedos:** se suman los dedos de las dos manos (1–10) y suena ese grado de la escala de Do.

**Rendimiento:** de unos 450 ms a unos 150 ms desde el gesto hasta el sonido (estimado a 30 fps). Así se logró:
- Cámara a 640×480.
- Procesamiento de cada cuadro apenas llega (`requestVideoFrameCallback`).
- Estabilización de 2 cuadros antes de cambiar de acorde.
- Ataque del sonido de 30 ms.

**Para la TV:** el diseño ocupa una pantalla de 16:9 sin scroll y se escala solo en 1080p y 4K. Atajos: `F` pantalla completa, `H` ocultar controles, `Espacio` silenciar.

## 7. Cómo ejecutar

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt          # requiere Python 3.9–3.12 por mediapipe

python extract_landmarks.py              # 1. dataset -> landmarks.npz
python train_ddp.py --rank 0 --world-size 1 --master-addr 127.0.0.1   # 2a. entrenar en una PC
# 2b. o en 2 PCs: ver PASOS_ENTRENAMIENTO.md
python export_web_model.py               # 3. hand_model.pt -> web/model.json
python -m http.server 8000 --directory web   # 4. dashboard en http://localhost:8000
```

Herramientas de apoyo:
- `python count_fingers.py --eval` mide las reglas de conteo sobre el dataset.
- `python count_fingers.py` abre el contador de dedos con la webcam.
- `python realtime.py` abre la demo con webcam del MLP en Python.

## 8. Estructura

```
dataset_unificado/       imágenes por clase
common.py                clases, normalización, MLP, augmentación
extract_landmarks.py     imágenes -> landmarks.npz (MediaPipe + CLAHE)
train_ddp.py             entrenamiento distribuido (DDP, gloo)
realtime.py              demo webcam del MLP (Python)
count_fingers.py         conteo de dedos por reglas (Python)
export_web_model.py      hand_model.pt -> web/model.json
PASOS_ENTRENAMIENTO.md   guía paso a paso del entrenamiento en 2 PCs
web/                     dashboard (index.html, style.css, js/)
```
