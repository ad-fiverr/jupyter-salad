# Jupyter Salad

Imagen independiente de JupyterLab para Salad Container Engine, basada en la imagen oficial de RunPod PyTorch. El repositorio contiene todo lo necesario para validar, construir y publicar la imagen desde GitHub Actions.

## Componentes

| Componente | Versión o configuración |
| --- | --- |
| Imagen base | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| Digest base | `sha256:61a4aafb0094cd773f11eefa378929d5a687bd775febeb78eac62fc824141fb5` (`linux/amd64`) |
| Python | 3.11 |
| PyTorch | 2.4.1+cu124 (contenido observado en el digest fijado) |
| CUDA runtime / toolkit | 12.4 / 12.4.1 |
| JupyterLab | 4.6.4 |
| Notebook | 7.6.3 |
| Jupyter Server | 2.21.1 |
| ipywidgets / JupyterLab widgets | 8.1.9 / 3.0.17 |
| Jupyter Server Terminals | 0.5.4 |
| IPython kernel | 7.3.0 |

El digest de la imagen base está fijado. Los paquetes Jupyter directos tienen versiones fijadas en `requirements-jupyter.txt`; sus dependencias transitivas se resuelven durante el build, por lo que esto no afirma reproducibilidad bit a bit. PyTorch, Python y CUDA se heredan de la imagen base.

El tag de RunPod conserva `2.4.0` en su nombre, pero el runtime que contiene el digest fijado informa `torch.__version__ == 2.4.1+cu124`. El verificador exige la versión base `2.4.1` (ignorando el sufijo local `+cu124`) y CUDA de PyTorch `12.4.x`; no instala ni degrada PyTorch durante el build. La etiqueta publicada de esta imagen refleja el runtime observado: `2.4.1-py3.11-cuda12.4.1`.

`jupyter_server_terminals==0.5.4` incluye un fragmento de configuración que habilita la extensión automáticamente. El Dockerfile no ejecuta `jupyter server extension enable --py`; el verificador comprueba que el fragmento instalado la habilita y que `jupyter server extension list` la descubre y valida como `OK`.

Se conserva el `ENTRYPOINT` de NVIDIA heredado de la imagen base. El Dockerfile reemplaza únicamente `CMD` para iniciar JupyterLab y omitir `/start.sh` de RunPod mientras mantiene la inicialización del runtime NVIDIA.

## Salad Container Group

Configura el Container Gateway y la aplicación con estos valores:

| Ajuste | Valor |
| --- | --- |
| Gateway port | `8888` |
| Protocol | HTTP |
| Dirección de escucha | IPv4 `0.0.0.0` e IPv6 `[::]` |
| Secreto de runtime | `JUPYTER_PASSWORD`, con una contraseña larga y única |
| Startup/readiness | HTTP `GET /login` en el puerto `8888` |
| Autenticación del Gateway | Deshabilitada para permitir WebSockets del navegador; Jupyter conserva su autenticación por contraseña |
| Liveness | Déjalo sin configurar inicialmente |

`nginx-main.conf` reemplaza la configuración principal heredada de RunPod. Define `events`/`http` e incluye únicamente `/etc/nginx/conf.d/default.conf`; no hereda servidores o includes adicionales. No se requiere `mime.types`: esta imagen sólo proxifica respuestas HTTP/WebSocket y no sirve archivos estáticos locales. `nginx-salad.conf` escucha en IPv4 `0.0.0.0:8888` e IPv6 `[::]:8888` (`ipv6only=on`) para conservar el acceso del Gateway de Salad. El Docker HEALTHCHECK y las pruebas funcionales de CI consultan `http://127.0.0.1:8888/login` con bypass de proxy; así CI no depende del soporte de loopback IPv6 de Docker. Las probes directas a Jupyter (`127.0.0.1:8889`) y ASR (`127.0.0.1:8765`) se mantienen. El build ejecuta `nginx -t` y un contrato sobre `nginx -T` que exige la configuración principal propia, el único include/server esperado y ambos listeners dentro del mismo bloque `server`.

JupyterLab usa `/workspace` como directorio de trabajo y raíz del servidor. El acceso sin autenticar está deshabilitado y los tokens de Jupyter también; la contraseña `JUPYTER_PASSWORD` es obligatoria. El servidor envía pings WebSocket cada 30 segundos para mantenerse por debajo del timeout de inactividad documentado por Salad. Si el puerto `8888` está ocupado, el servidor falla en vez de cambiar de puerto.

Jupyter permite ejecutar código con la identidad del usuario autenticado. No despliegues la imagen sin establecer `JUPYTER_PASSWORD` como secreto de runtime. `.env.example` contiene únicamente un marcador y no debe usarse como contraseña real.

El contenedor no configura almacenamiento persistente. El contenido de `/workspace`, incluidos notebooks, modelos, checkpoints y resultados, puede perderse al reemplazar o eliminar el contenedor. Copia los datos importantes a almacenamiento persistente antes de terminar el trabajo.

## Build y publicación

Repositorio: [ad-fiverr/jupyter-salad](https://github.com/ad-fiverr/jupyter-salad)
Imagen: [myblockchaincompany/jupyter-salad en Docker Hub](https://hub.docker.com/repository/docker/myblockchaincompany/jupyter-salad/general)

El workflow [`.github/workflows/build.yml`](.github/workflows/build.yml) se ejecuta al hacer push a `main` cuando cambia un archivo de build o smoke test y también permite `workflow_dispatch`. Antes del build valida YAML/shell y ejecuta tests sin Docker para fases/diagnósticos, bypass de proxy, configuración efectiva y listeners nginx IPv4/IPv6. Usa el contexto `.` y `./Dockerfile`, prepara Buildx y construye `linux/amd64`. Después ejecuta el smoke test sobre la imagen cargada usando IPv4 loopback. El login a Docker Hub y los pushes ocurren únicamente si el build y el smoke test pasan.

### Diagnóstico de nginx en CI

La evidencia del [run real de GitHub Actions](https://github.com/ad-fiverr/jupyter-salad/actions/runs/36724378654/job/109917342589) identificó la causa: `nginx -T` mostró que la configuración heredada de RunPod no incluía `/etc/nginx/conf.d/default.conf` ni listeners 8888, aunque el archivo de servidor en disco sí declaraba ambos. El main heredado exponía servidores en 9091, 3001, 7861, 8081, 8001 y 7270. Por tanto, `nginx -t` pasaba sobre una configuración válida pero ajena a Salad, y el contrato efectivo fallaba. `ROOT_CAUSE=RUNPOD_NGINX_MAIN_CONFIG_DID_NOT_INCLUDE_OUR_CONF`. Esta revisión añade un main config controlado por el repo y refuerza el contrato; los tests locales usan fixtures. El comportamiento real de la imagen corregida aún requiere un build de GitHub Actions.

Ante timeout, unhealthy o salida del contenedor, el smoke conserva el diagnóstico previo y añade, dentro del mismo contenedor:

- `nginx -T`: solo markers de origen, ruta principal, includes y bloques/listeners; lectura de `nginx.conf` y `default.conf` identificada por separado como evidencia de disco.
- Flags `NGINX_EFFECTIVE_CONFIG_CAPTURED`, `NGINX_CONF_D_INCLUDED`, `NGINX_EFFECTIVE_IPV4_8888` y `NGINX_EFFECTIVE_IPV6_8888`, derivados del dump efectivo; los archivos en disco no pueden satisfacer el contrato.
- Todos los listeners TCP de `ss -ltnp` cuando está disponible y tablas `/proc/net/tcp{,6}` para confirmar familias/puertos, incluido cualquier puerto distinto de 8888; no instala paquetes. Estado de procesos nginx y override `-c` si existe, sin volcar argumentos completos.
- Flags `RUNTIME_TCP_8888_IPV4`, `RUNTIME_TCP_8888_IPV6`, `JUPYTER_8889` y `ASR_8765` derivados de sockets en estado LISTEN en `/proc`; `UNAVAILABLE` significa que la tabla no se pudo leer.
- Cuatro probes TCP puros antes de HTTP: IPv4 8888/8889/8765 e IPv6 `::1:8888`. Solo nombre, resultado, errno y clase de excepción; IPv6 sin soporte queda `UNAVAILABLE`.

Cada llamada Docker de diagnóstico mantiene el límite de 5 s y los probes de socket usan 2 s; el loop de health de 45 × 2 s se conserva. No se imprimen dumps completos de configuración ni errores crudos que puedan contener credenciales. `nginx -T` describe la configuración efectiva en disco en el momento de la prueba, no permite leer la configuración que un master antiguo guarda en memoria. El nuevo contrato estricto comprueba el main config propio, el único include/server de Salad y los dos listeners. Sus fixtures unitarios pasan localmente; la ejecución de `nginx -t`/`nginx -T` dentro de la imagen corregida y la captura real de sockets quedan pendientes del próximo run de GitHub Actions.

Configura estos GitHub Actions Secrets en el repositorio:

- `DOCKERHUB_USERNAME`: usuario con permiso de escritura en `myblockchaincompany/jupyter-salad`.
- `DOCKERHUB_TOKEN`: access token de Docker Hub.

El workflow publica estos tags desde la misma imagen que pasó el smoke test:

- `myblockchaincompany/jupyter-salad:2.4.1-py3.11-cuda12.4.1`
- `myblockchaincompany/jupyter-salad:latest`
- `myblockchaincompany/jupyter-salad:sha-<short-commit>`

El runner libera espacio y exige al menos 20 GiB disponibles antes del build; la imagen base ocupa aproximadamente 6.92 GB comprimida. No se necesita `HF_TOKEN` ni una credencial de Salad para construir y publicar esta imagen.

El workflow ya fija `platforms: linux/amd64`; el Dockerfile deja que Buildx aplique esa plataforma a `FROM`, evitando un `--platform` constante redundante.

## Estado de validación

| Validación | Estado |
| --- | --- |
| Build Docker local | `LOCAL_DOCKER_BUILD = UNRUN` — Build intentionally delegated to GitHub Actions per user instruction. |
| Primer build en GitHub Actions | `FAIL` — [run #1](https://github.com/ad-fiverr/jupyter-salad/actions/runs/36597467763): `notebook==7.6.3` requiere `jupyterlab>=4.6.4,<4.7`; estaba fijado `jupyterlab==4.6.3`. |
| Segundo build en GitHub Actions | `FAIL` — [run #2](https://github.com/ad-fiverr/jupyter-salad/actions/runs/36602803603): instalación Jupyter completada; el verificador falló porque esperaba PyTorch 2.4.0 y la base contiene `2.4.1+cu124`. `jupyter_server extension enable` también informó que no encontraba el módulo, aunque su comando devolvió éxito. |
| Tercer build en GitHub Actions | `FAIL` — [run #3](https://github.com/ad-fiverr/jupyter-salad/actions/runs/36608666952): falló el verificador al acceder a `jupyter_server.ServerApp`; esa clase está en `jupyter_server.serverapp`. La instalación de pip sí terminó. |
| Cuarto build en GitHub Actions | `FAIL` — [run #4](https://github.com/ad-fiverr/jupyter-salad/actions/runs/36609873658): el listado muestra `jupyter_server_terminals enabled` y validación `OK`, pero el verificador no reconoce la línea como habilitada; se normaliza la salida ANSI y se validan sus tokens. |
| Quinto build en GitHub Actions | `FAIL` — [run #5](https://github.com/ad-fiverr/jupyter-salad/actions/runs/36611124163): el contenedor pasó el healthcheck IPv6, pero la primera petición del smoke test desde el runner a `127.0.0.1` terminó en `ConnectionResetError`; en esa revisión, el smoke del contenedor probó luego el listener por `[::1]:8888`. Un run posterior mostró que ese probe IPv6 loopback también falla en el Docker runner ([run #36680271147](https://github.com/ad-fiverr/jupyter-salad/actions/runs/36680271147)); la ruta funcional actual de CI usa `127.0.0.1:8888`, mientras Salad conserva el listener público IPv6. |
| Reintento con esta corrección | `GITHUB_ACTIONS_RETEST = UNRUN` — awaiting user push/re-run. |
| Validaciones estáticas tras la corrección del run #5 | `PASS` — sintaxis Bash y Python embebido, parseo YAML, `git diff --check`, paths, versiones, invariantes de runtime, orden build/smoke/login/push y escaneo de secretos del worktree. |
| Publicación Docker Hub | `DOCKERHUB_PUBLICATION = UNRUN` |
| Despliegue real en Salad | `SALAD_REAL_DEPLOYMENT = UNRUN` |
| Prueba real con GPU / entrenamiento | `GPU_RUNTIME_TEST = UNRUN` |

## Referencias

- [RunPod PyTorch 2.4 y CUDA 12.4](https://www.runpod.io/articles/guides/pytorch-2-4-cuda-12-4)
- [Manifest fijado de RunPod en Docker Hub](https://hub.docker.com/layers/runpod/pytorch/2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04/images/sha256-61a4aafb0094cd773f11eefa378929d5a687bd775febeb78eac62fc824141fb5)
- [Red e IPv6 de Salad](https://docs.salad.com/container-engine/explanation/infrastructure-platform/networking)
- [WebSockets de Salad](https://docs.salad.com/container-engine/explanation/gateway/websockets)
- [Health probes de Salad](https://docs.salad.com/container-engine/explanation/infrastructure-platform/health-probes)
- [Seguridad de Jupyter Server](https://jupyter-server.readthedocs.io/en/latest/operators/security.html)
- [Configuración de Jupyter Server](https://jupyter-server.readthedocs.io/en/stable/other/full-config.html)
- [Cómo Jupyter Server descubre y valida extensiones](https://jupyter-server.readthedocs.io/en/stable/operators/multiple-extensions.html)
- [jupyter_server_terminals 0.5.4 en PyPI](https://pypi.org/project/jupyter-server-terminals/0.5.4/)
- [Configuración auto-enable incluida en la etiqueta 0.5.4](https://github.com/jupyter-server/jupyter_server_terminals/blob/v0.5.4/jupyter-config/jupyter_server_terminals.json)
- [PyTorch 2.7 y soporte para Blackwell](https://pytorch.org/blog/pytorch-2-7/)

## ASR Lab (experimental)

This image now includes an optional ASR laboratory behind the same Salad Container Gateway. The public listener remains IPv6 port `8888`; nginx routes `/` to password-protected JupyterLab at `127.0.0.1:8889` and `/asr/*` to FastAPI at `127.0.0.1:8765`. Only `8888` is intended for the Container Gateway.

Set these in the Salad Container Group as runtime values:

| Variable | Value |
| --- | --- |
| `JUPYTER_PASSWORD` | Required unique password Secret |
| `ASR_BACKEND` | Exactly one: `parakeet` or `faster_whisper` |
| `ASR_API_TOKEN` | Unique random Secret, minimum 24 bytes; use at least 32 random bytes |
| `ASR_WORKERS` | Any positive integer; each worker loads its own full model, with no configured hard maximum |
| `HF_TOKEN` | Optional Salad Secret only when Hugging Face access requires it |
| `SALAD_FULL_RUNTIME_VERIFY` | Optional debug flag, default `0`; set to `1` to repeat the full Python/Torch/Jupyter/NeMo verifier before services start. The complete verifier always runs during the image build; normal startup checks package metadata only. |

There is no image default backend. `parakeet` uses the previously tested NVIDIA NeMo runtime (`nemo_toolkit[asr]==2.4.0`) and the historical RMS/blacklist behavior. Changing `ASR_BACKEND` requires restarting the container; the unselected backend is not loaded. Parakeet jobs are distributed round-robin among workers. More workers increase concurrency/throughput, not the speed of one inference; VRAM fit and practical limits must be measured on a real GPU. `ASR_API_TOKEN` is accepted as `?token=` for the browser WebSocket client. This is a lab credential, so avoid sharing the URL and use a dedicated token. ASR and nginx access logs are disabled; the Jupyter process does not inherit the ASR/Hugging Face secrets. The full Jupyter/NeMo import checks run at image build; ordinary container starts use lightweight package-metadata checks so nginx and Jupyter can start without waiting for a second NeMo import.

Public paths:

- Jupyter UI: `https://<salad-host>/`
- ASR WebSocket: `wss://<salad-host>/asr/ws?token=<ASR_API_TOKEN>`
- ASR process health: `https://<salad-host>/asr/health`
- ASR model readiness: `https://<salad-host>/asr/readiness` (503 until model warm-up finishes)

The ASR protocol is mono PCM16LE at 16 kHz, base64 encoded, compatible with the existing MyBrainAssistant message. Segments are finalized by server silence/inactivity handling; the API does not emit partial-token transcripts. A full reproducible guide and honest test status are maintained in the C2C workspace at `ASR/ASR_SALAD_LAB.md`.

The image build and runner smoke test remain in GitHub Actions. This implementation was not built locally, published, or deployed to Salad. Do not point a Group that is actively training at a new image until you have saved its work and intentionally scheduled a separate test.

### Live ASR Benchmark

Open `https://<salad-host>/asr/benchmark` while the selected ASR backend is ready. The page requests microphone access, converts the live stream to mono PCM16LE at 16 kHz in 100 ms chunks with `AudioWorklet`, and uses the existing authenticated `/asr/ws` protocol plus its existing `flush` barrier. It shows final transcripts by segment (`TRANSCRIPT_MODE=FINAL_SEGMENT`); it does not claim partial or word-level streaming.

Enter `ASR_API_TOKEN` in the password field. The page keeps it in memory only, sends it to `/asr/telemetry` as a Bearer header and to `/asr/ws` using the existing query-token contract, and clears it when the run ends. The websocket and nginx access logs remain disabled. The page does not write audio to disk or browser storage; it releases microphone tracks at stop. JSON/CSV exports contain experiment metadata, transcript, timing samples and 1 Hz telemetry, but no token or audio. CSV text cells neutralize spreadsheet formulas.

The telemetry endpoint is authenticated and allowlists backend/model, readiness, configured worker count, queued-job depth, process RSS, `/proc/meminfo` RAM, and GPU0 utilization, VRAM, temperature and power when the optional `pynvml` binding is available. NVML is not a required ASR dependency; unavailable GPU measurements remain `null`. Polling is 1 Hz, not per audio chunk. `CLIENT_AUDIO_END_TO_TRANSCRIPT_MS` uses the browser's compatible shadow RMS gate and one stable per-segment request ID; its resolution includes 100 ms chunking and browser scheduling. `SERVER_AUDIO_END_TO_TRANSCRIPT_MS` uses only the server monotonic clock and includes server silence gating, queue wait and inference until the transcript is ready. These are distinct measurements and are not partial-token latency. The canonical formulas, limitations, and word-timestamp finding are documented in `voice_agent_llm/ASR/ASR_SALAD_LAB.md`.

Qwen streaming is a separate experimental build/runtime switch: build with `INSTALL_QWEN_RUNTIME=1`, then set `QWEN_STREAMING_ENABLED=1` and `QWEN_MAX_ACTIVE_STREAMS=6` in Salad. It still uses one Qwen engine/process and one decode RPC at a time; six is the logical stream limit, not a claim of GPU capacity. Its bounded round-robin scheduler reports queue wait, pending jobs, audio backlog, stream lag and decode wall, and fences only a stream that disconnects or overruns its configured backlog. `--concurrency-matrix` in `asr-lab/benchmark/benchmark_ws.py` compares 1/2/4/6 fixture runs; actual throughput, VRAM fit and chunk cadence remain unverified until a real Salad GPU measurement. The dashboard hides batch-only latency fields in Qwen mode and caps mobile canvas backing dimensions.