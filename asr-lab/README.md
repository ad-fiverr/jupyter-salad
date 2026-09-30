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

`GET /asr/telemetry` exige `Authorization: Bearer <ASR_API_TOKEN>`. El esquema permite únicamente backend/modelo, readiness/workers/cola, RSS/RAM y GPU; no devuelve variables de entorno. Si `pynvml` está disponible en la imagen, NVML se lee una vez por muestra de 1 Hz; no es una dependencia obligatoria del runtime ASR. Si falta el binding, no hay device/driver o un campo no está soportado, ese dato queda `null`. El audio permanece en memoria y se libera al terminar; token y micrófono no se guardan en browser storage ni exportaciones.

`SERVER_AUDIO_END_TO_TRANSCRIPT_MS` se calcula con `time.perf_counter()` desde el último chunk que el VAD servidor clasificó como voz hasta que el transcript está listo; incluye el cierre por silencio, cola e inferencia, no la latencia del Gateway. `CLIENT_AUDIO_END_TO_TRANSCRIPT_MS` se mide con `performance.now()` desde el último chunk browser sobre RMS 0.015 hasta recibir el transcript, correlacionado por request ID; su resolución es aproximada a un chunk más scheduling del navegador. No se comparan relojes cliente/servidor. Los exports JSON/CSV incluyen las muestras; el CSV protege celdas que podrían interpretarse como fórmulas. Los timestamps por palabra son soportados por la API upstream de NeMo, pero el adapter actual no los solicita y su funcionamiento en el NeMo fijado no se verificó: no se habilitan en esta tarea.

La UI y sus assets se validan con tests Node sin dependencias de navegador. El Docker build verifica que todos los assets estén en la imagen y corre la suite Python usando fakes; no descarga pesos ni requiere GPU.
