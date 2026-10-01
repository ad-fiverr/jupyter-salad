# ASR Salad Lab

Implementación canónica del laboratorio de voz. Sirve al contenedor Jupyter Salad por `WS /asr/ws`. El backend se elige explícitamente con `ASR_BACKEND=parakeet|faster_whisper`. Parakeet usa el adapter histórico de NVIDIA NeMo; cada worker carga su propia instancia de modelo. `ASR_WORKERS` acepta cualquier entero positivo y distribuye jobs por round-robin. Los workers aumentan concurrency/throughput; no dividen una inferencia ni garantizan que todas las instancias quepan en VRAM.

Véase [ASR_SALAD_LAB.md](../ASR_SALAD_LAB.md) para arquitectura, compatibilidad, seguridad, Salad, fixture corpus, métricas y estados de pruebas.

Los modelos descargan en runtime bajo `/workspace/.cache/huggingface`; no se versionan pesos. `ASR_API_TOKEN` y, cuando sea necesario, `HF_TOKEN` se configuran como Salad Secrets.

Validaciones locales disponibles sin Docker:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m unittest discover -s tests -v
python -m compileall -q asr_lab benchmark tests
```

El build local/Salad y el benchmark GPU requieren el flujo autorizado del usuario y están `UNRUN`.

## Live ASR Benchmark

La página del laboratorio queda en `GET /asr/benchmark`. Usa `getUserMedia`, `AudioContext` y `AudioWorklet` para emitir mono PCM16LE a 16 kHz en bloques de 100 ms por el WebSocket existente `/asr/ws`; detener envía `flush` y espera `flush_complete`. La UI sólo presenta transcripciones finales por segmento (`TRANSCRIPT_MODE=FINAL_SEGMENT`).

`GET /asr/telemetry` exige `Authorization: Bearer <ASR_API_TOKEN>`. El esquema permite únicamente backend/modelo, readiness/workers/cola, RSS/RAM y GPU; no devuelve variables de entorno. NVML es opcional y se lee a 1 Hz. Si falta o falla, Torch puede aportar dispositivo y VRAM global; utilización/temperatura/potencia quedan `null`. Las mediciones NVML/Torch son globales al dispositivo, no atribuibles al backend. El allocator de Torch no representa VRAM de CTranslate2. El audio permanece en memoria y se libera al terminar; token y micrófono no se guardan en browser storage ni exportaciones.

El benchmark separa `AUDIO_DURATION_MS` (duración de PCM entregado, no latencia), `SERVER_ENDPOINTING_MS`, `SERVER_QUEUE_WAIT_MS`, `SERVER_MODEL_INFERENCE_MS`, `SERVER_POSTPROCESS_MS` y `SERVER_EOS_TO_TRANSCRIPT_MS`. Los cuatro componentes del servidor usan `time.perf_counter()` del mismo proceso y suman aproximadamente el EOS→transcript. `SERVER_MODEL_INFERENCE_MS` es wall time del adaptador `backend.transcribe()`, incluida la materialización del resultado Faster-Whisper; no mide kernel GPU. `CLIENT_EOS_TO_TRANSCRIPT_MS` usa sólo `performance.now()` y cruza navegador, red, proxy, servidor y regreso. `PROXY_WS_RTT_MS` es un ping/pong JSON de baja frecuencia por el mismo WebSocket autenticado; es RTT completo, no latencia unidireccional. No se mezclan percentiles cliente y servidor.

`SERVER_RECEIVE_TO_TRANSCRIPT_MS` se conserva sólo como diagnóstico desde el inicio del buffer y puede incluir silencio inicial y duración hablada; no es KPI de latencia. Los nombres anteriores `MODEL_INFERENCE_MS`, `SEGMENT_WAIT_MS`, `queue_wait_ms`, `SERVER_AUDIO_END_TO_TRANSCRIPT_MS` y `CLIENT_AUDIO_END_TO_TRANSCRIPT_MS` siguen como aliases documentados. JSON incluye `metric_definitions` y muestras RTT separadas; CSV incluye métricas por segmento y filas de RTT diferenciadas. El CSV protege celdas que podrían interpretarse como fórmulas. Los timestamps por palabra son soportados por la API upstream de NeMo, pero el adapter actual no los solicita y su funcionamiento en el NeMo fijado no se verificó: no se habilitan en esta tarea.

La UI y sus assets se validan con tests Node sin dependencias de navegador. El Docker build verifica que todos los assets estén en la imagen y corre la suite Python usando fakes; no descarga pesos ni requiere GPU.
