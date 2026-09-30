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
