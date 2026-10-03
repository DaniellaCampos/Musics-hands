# Entrenamiento en paralelo con 2 computadoras

Modelo que reconoce el gesto de **cada mano (hasta 2)** frente a la cámara: 1–5 dedos, rock y pinch.
Si las dos manos muestran un número, también muestra la suma.

## Cómo funciona

1. **MediaPipe Hands** detecta hasta 2 manos y entrega 21 puntos (x, y, z) por mano.
2. Los puntos se normalizan (centrados en la muñeca y escalados), así que da igual dónde esté la mano o a qué distancia.
3. Un **MLP pequeño** (63 → 128 → 64 → 7) clasifica cada mano.
4. El entrenamiento usa **PyTorch DistributedDataParallel (DDP)** con backend `gloo`:
   - cada PC entrena con **la mitad** de los datos de entrenamiento (`DistributedSampler`);
   - en cada paso, las dos PCs **promedian sus gradientes** por la red, así los dos modelos quedan idénticos.
5. Como las PCs están en **redes distintas**, se conectan con **Tailscale**, que crea una red privada virtual entre ellas.

> El entrenamiento corre en CPU. El modelo es muy pequeño y la GPU no lo acelera; además, `gloo` en Windows funciona mejor con CPU.

| Archivo | Qué hace |
|---|---|
| `common.py` | Clases, normalización de landmarks, modelo y augmentación |
| `extract_landmarks.py` | Dataset de imágenes → `landmarks.npz` |
| `train_ddp.py` | Entrenamiento distribuido → `hand_model.pt` |
| `realtime.py` | Demo con webcam (hasta 2 manos) |

---

## Paso 1: Preparar el entorno (en las 2 PCs)

```powershell
git clone <url-del-repo> Musics-hands
cd Musics-hands
python -m venv .venv
.\.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -c "import torch; print(torch.__version__)"
```

**Las dos PCs deben tener la misma versión de torch.** Si no coinciden, el entrenamiento se queda colgado o falla.

## Paso 2: Conectar las PCs con Tailscale (en las 2 PCs)

1. Instalen Tailscale desde https://tailscale.com/download/windows.
2. Inicien sesión con **la misma cuenta** en las dos PCs.
3. En la **PC maestra** (rank 0), anoten su IP:
   ```powershell
   tailscale ip -4        # ej. 100.101.102.103
   ```
4. Desde la otra PC, comprueben que la ven:
   ```powershell
   ping 100.101.102.103
   ```
   Si el ping no responde, el firewall de Windows está bloqueando ICMP. Puede ser normal: hagan el Paso 3 y prueben igual.

## Paso 3: Configurar el firewall (en las 2 PCs, PowerShell como administrador)

`gloo` usa el puerto 29500 y además abre puertos aleatorios entre las PCs. Por eso hay que permitir el **ejecutable de Python del venv**, no solo un puerto:

```powershell
New-NetFirewallRule -DisplayName "Python DDP" -Direction Inbound -Action Allow `
  -Program "$PWD\.venv\Scripts\python.exe" -Profile Any
```

Además, si Windows muestra el aviso "¿Permitir acceso?" la primera vez que se ejecuta, acepten **redes privadas y públicas**.

## Paso 4: Indicar a gloo qué red usar (en las 2 PCs, en cada terminal nueva)

```powershell
Get-NetAdapter              # busquen el adaptador "Tailscale"
$env:GLOO_SOCKET_IFNAME = "Tailscale"
```

Sin esto, `gloo` puede elegir la WiFi local y la otra PC no podrá conectarse.

## Paso 5: Extraer landmarks (solo una vez, en la PC maestra)

```powershell
python extract_landmarks.py
```

Muestra el % de imágenes donde se detectó la mano, por clase. Las que fallan se listan en `sin_deteccion.txt`.
Suban el resultado para que las dos PCs usen **exactamente los mismos datos**:

```powershell
git add landmarks.npz
git commit -m "Agregar landmarks extraídos"
git push
```

En la otra PC: `git pull`.

## Paso 6: Entrenar en paralelo

**PC maestra (rank 0):**
```powershell
python train_ddp.py --rank 0 --world-size 2 --master-addr 100.101.102.103 --master-port 29500
```

**PC 2 (rank 1)**, con la IP de la **maestra**:
```powershell
python train_ddp.py --rank 1 --world-size 2 --master-addr 100.101.102.103 --master-port 29500
```

Rank 0 se queda esperando hasta que rank 1 se conecta (máximo 5 minutos). Después, cada PC muestra lo siguiente:

```
[rank 0 | PC-MIGUEL] train total=840, mi parte=420, val=210
[rank 1 | PC-OTRA]   train total=840, mi parte=420, val=210
[rank 0 | PC-MIGUEL] epoch  10  loss 0.31  val_acc(global) 0.952  muestras vistas por mí 4200
```

En la demo, señalen que:
- `mi parte=420`: cada PC procesa solo la mitad de los datos;
- `val_acc(global)` es **igual en las dos PCs**, porque comparten el mismo modelo, sincronizado en cada paso.

Al terminar, la PC maestra guarda `hand_model.pt` y muestra la matriz de confusión.

**Comparación con una sola PC (opcional):**
```powershell
python train_ddp.py --rank 0 --world-size 1 --master-addr 127.0.0.1
```

> Con este dataset tan pequeño, 2 PCs por internet **no serán más rápidas** que una: la comunicación tarda más que el cálculo. Lo que se demuestra es el mecanismo. Con datasets o modelos grandes, la ganancia sí es real.

## Paso 7: Probar con la webcam (en la PC maestra o copiando `hand_model.pt`)

```powershell
python realtime.py --model hand_model.pt
```

- Cada mano muestra `Izq/Der: <gesto> (confianza)`.
- Si las dos manos muestran números (1–5), arriba aparece `Total: N`.
- Pulsen `q` para salir. Con `--camera 1` se cambia de cámara y con `--threshold 0.7` se exige más confianza.

---

## Ensayo previo en una sola PC

Antes de la demo, simulen las 2 PCs con dos terminales en la misma máquina:

```powershell
# Terminal 1
python train_ddp.py --rank 0 --world-size 2 --master-addr 127.0.0.1
# Terminal 2
python train_ddp.py --rank 1 --world-size 2 --master-addr 127.0.0.1
```

En este ensayo **no** definan `GLOO_SOCKET_IFNAME`.

## Problemas típicos

| Síntoma | Solución |
|---|---|
| Se queda en "conectando..." y luego timeout | IP incorrecta, firewall (Paso 3), o falta `GLOO_SOCKET_IFNAME` (Paso 4). Revisen `tailscale status`. |
| `use_libuv was requested but PyTorch was built without libuv` | Ya se resuelve en el script con `USE_LIBUV=0`. Si aparece, ejecuten `$env:USE_LIBUV="0"`. |
| Se cuelga después de conectar | Versiones distintas de torch, o `landmarks.npz` distinto entre las PCs. |
| Aviso `client socket has failed to connect to [kubernetes.docker.internal]` | Es inofensivo (lo provoca Docker Desktop en el archivo hosts). Si después aparece "conectado al grupo", todo va bien. |
| `Address already in use` | Quedó un proceso anterior abierto. Ciérrenlo o usen `--master-port 29501` en las dos PCs. |
| Pocas imágenes con mano detectada | `python extract_landmarks.py --min-detection-confidence 0.2` |
| La webcam confunde 3 y 4 dedos | Pongan la mano de frente a la cámara, con buena luz, o suban `--threshold`. |
