# Jupyter Salad

Imagen independiente de JupyterLab para Salad Container Engine, basada en la imagen oficial de RunPod PyTorch. El repositorio contiene todo lo necesario para validar, construir y publicar la imagen desde GitHub Actions.

## Componentes

| Componente | Versión o configuración |
| --- | --- |
| Imagen base | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| Digest base | `sha256:61a4aafb0094cd773f11eefa378929d5a687bd775febeb78eac62fc824141fb5` (`linux/amd64`) |
| Python | 3.11 |
| PyTorch | 2.4.0 |
| CUDA runtime / toolkit | 12.4 / 12.4.1 |
| JupyterLab | 4.6.3 |
| Notebook | 7.6.3 |
| Jupyter Server | 2.21.1 |
| ipywidgets / JupyterLab widgets | 8.1.9 / 3.0.17 |
| Jupyter Server Terminals | 0.5.4 |
| IPython kernel | 7.3.0 |

El digest de la imagen base está fijado. Los paquetes Jupyter directos tienen versiones fijadas en `requirements-jupyter.txt`; sus dependencias transitivas se resuelven durante el build, por lo que esto no afirma reproducibilidad bit a bit. PyTorch, Python y CUDA se heredan de la imagen base.

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

- `myblockchaincompany/jupyter-salad:2.4.0-py3.11-cuda12.4.1`
- `myblockchaincompany/jupyter-salad:latest`
- `myblockchaincompany/jupyter-salad:sha-<short-commit>`

El runner libera espacio y exige al menos 20 GiB disponibles antes del build; la imagen base ocupa aproximadamente 6.92 GB comprimida. No se necesita `HF_TOKEN` ni una credencial de Salad para construir y publicar esta imagen.

## Estado de validación

| Validación | Estado |
| --- | --- |
| Build Docker local | `LOCAL_DOCKER_BUILD = UNRUN` — Build intentionally delegated to GitHub Actions per user instruction. |
| GitHub Actions build y smoke test | `GITHUB_ACTIONS_BUILD = UNRUN` hasta ejecutar el workflow en GitHub |
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
- [PyTorch 2.7 y soporte para Blackwell](https://pytorch.org/blog/pytorch-2-7/)
