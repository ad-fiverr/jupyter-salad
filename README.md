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
| Dirección de escucha | IPv6 `::` |
| Secreto de runtime | `JUPYTER_PASSWORD`, con una contraseña larga y única |
| Startup/readiness | HTTP `GET /login` en el puerto `8888` |
| Autenticación del Gateway | Deshabilitada para permitir WebSockets del navegador; Jupyter conserva su autenticación por contraseña |
| Liveness | Déjalo sin configurar inicialmente |

JupyterLab usa `/workspace` como directorio de trabajo y raíz del servidor. El acceso sin autenticar está deshabilitado y los tokens de Jupyter también; la contraseña `JUPYTER_PASSWORD` es obligatoria. El servidor envía pings WebSocket cada 30 segundos para mantenerse por debajo del timeout de inactividad documentado por Salad. Si el puerto `8888` está ocupado, el servidor falla en vez de cambiar de puerto.

Jupyter permite ejecutar código con la identidad del usuario autenticado. No despliegues la imagen sin establecer `JUPYTER_PASSWORD` como secreto de runtime. `.env.example` contiene únicamente un marcador y no debe usarse como contraseña real.

El contenedor no configura almacenamiento persistente. El contenido de `/workspace`, incluidos notebooks, modelos, checkpoints y resultados, puede perderse al reemplazar o eliminar el contenedor. Copia los datos importantes a almacenamiento persistente antes de terminar el trabajo.

## Build y publicación

Repositorio: [ad-fiverr/jupyter-salad](https://github.com/ad-fiverr/jupyter-salad)
Imagen: [myblockchaincompany/jupyter-salad en Docker Hub](https://hub.docker.com/repository/docker/myblockchaincompany/jupyter-salad/general)

El workflow [`.github/workflows/build.yml`](.github/workflows/build.yml) se ejecuta al hacer push a `main` cuando cambia un archivo de build y también permite `workflow_dispatch`. Usa el contexto `.` y `./Dockerfile`, valida la sintaxis YAML y shell, prepara Buildx y construye `linux/amd64`. Después ejecuta el smoke test sobre la imagen cargada. El login a Docker Hub y los pushes ocurren únicamente si el build y el smoke test pasan.

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
| Quinto build en GitHub Actions | `FAIL` — [run #5](https://github.com/ad-fiverr/jupyter-salad/actions/runs/36611124163): el contenedor pasó healthcheck IPv6, pero la primera petición del smoke test desde el runner a `127.0.0.1` terminó en `ConnectionResetError`; ahora la prueba HTTP corre dentro del contenedor por `[::1]:8888`. |
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
