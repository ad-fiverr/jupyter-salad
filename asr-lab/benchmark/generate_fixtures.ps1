param(
  [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\fixtures')
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
$synth = [System.Speech.Synthesis.SpeechSynthesizer]::new()
$voice = $synth.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo } | Where-Object { $_.Name -eq 'Microsoft Sabina Desktop' } | Select-Object -First 1
if (-not $voice) { throw 'Microsoft Sabina Desktop (es-MX) is unavailable; choose an installed offline es-MX SAPI voice and document it.' }
$synth.SelectVoice($voice.Name)
$format = [System.Speech.AudioFormat.SpeechAudioFormatInfo]::new(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, 1)
$items = @(
  @{ Name='frase_corta'; Text='Hola, ¿me escuchas bien? Vamos a empezar con la prueba.' },
  @{ Name='numeros_fechas_horas'; Text='La reunión será el martes 14 de octubre a las 3:45 de la tarde. Necesito 27 cajas, 13 piezas azules y 2 cafés.' },
  @{ Name='nombre_direccion_ficticios'; Text='Mariana López vive en la calle Nopal número 248, colonia Las Palmas, en Guadalajara.' },
  @{ Name='spanglish'; Text='El deadline del sprint es mañana; sube el dashboard al shared drive y me mandas el feedback.' },
  @{ Name='frase_larga'; Text='Buenos días, equipo. Para la prueba de hoy vamos a revisar el inventario del taller, confirmar las piezas que llegaron, comparar el total con la factura y enviar un resumen antes de las cinco. Si detectamos una diferencia, anoten el número de orden y avísenme por el canal general.' }
)
foreach ($item in $items) {
  $wav = Join-Path $OutputDirectory ($item.Name + '.wav')
  $truth = Join-Path $OutputDirectory ($item.Name + '.txt')
  $synth.SetOutputToWaveFile($wav, $format)
  $synth.Speak($item.Text)
  $synth.SetOutputToNull()
  [System.IO.File]::WriteAllText($truth, $item.Text, [System.Text.UTF8Encoding]::new($false))
}
$synth.Dispose()
