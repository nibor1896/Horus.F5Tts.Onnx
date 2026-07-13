# Horus.F5Tts.Onnx

Run [**F5-TTS**](https://github.com/SWivid/F5-TTS) text-to-speech from .NET using **pure
[ONNX Runtime](https://onnxruntime.ai/)** — no Python, no PyTorch. Give it a short reference voice
clip and some text, get 24 kHz audio back. Runs on CPU or any GPU your ONNX Runtime build supports
(DirectML for any DX12 GPU, CUDA for NVIDIA).

> Part of the [Horus](https://github.com/nibor1896) project, split out as a standalone library
> because a clean .NET runner for F5-TTS didn't exist yet.

## Install

```sh
dotnet add package Horus.F5Tts.Onnx
```

The library only pulls in the ONNX Runtime **managed** API. Add a native runtime package to pick
where inference runs:

```sh
# CPU (works everywhere, slow for the ~1.3 GB transformer)
dotnet add package Microsoft.ML.OnnxRuntime

# any DirectX 12 GPU (NVIDIA / AMD / Intel) — recommended on Windows
dotnet add package Microsoft.ML.OnnxRuntime.DirectML

# NVIDIA CUDA
dotnet add package Microsoft.ML.OnnxRuntime.Gpu
```

## Get the models

You need the three ONNX files + `vocab.txt` from an F5-TTS ONNX export
([DakeQQ/F5-TTS-ONNX](https://github.com/DakeQQ/F5-TTS-ONNX)):

- `F5_Preprocess.onnx`
- `F5_Transformer.onnx`
- `F5_Decode.onnx`
- `vocab.txt`

A ready-made **German** export lives at
[nibor1896/F5-TTS-German-ONNX](https://huggingface.co/nibor1896/F5-TTS-German-ONNX). For other
languages, export your checkpoint with DakeQQ's tooling.

## Quick start

```csharp
using Horus.F5Tts.Onnx;

// Load once (heavy). Append a GPU provider via the optional session hook.
using var model = F5TtsModel.Load(
    "models/F5_Preprocess.onnx",
    "models/F5_Transformer.onnx",
    "models/F5_Decode.onnx",
    "models/vocab.txt",
    configureSession: o => o.AppendExecutionProvider_DML(0)); // or omit for CPU

// Reference voice: 24 kHz mono 16-bit PCM, plus its transcript.
var (referenceAudio, _) = WavAudio.ReadPcm16("reference.wav");

var result = model.Synthesize(
    referenceAudio,
    referenceText: "This is the transcript of the reference clip.",
    text: "And this is the new sentence to speak.",
    new F5TtsOptions { Speed = 1.1f });

File.WriteAllBytes("out.wav", result.ToWav()); // 24 kHz mono WAV
```

`Synthesize` is synchronous and CPU/GPU-bound — call it from a background thread
(`await Task.Run(() => model.Synthesize(...))`) in UI apps.

## Notes

- **Reference audio** must be 24 kHz mono. `WavAudio.ReadPcm16` loads 16-bit PCM WAV (and
  down-mixes stereo) but does **not** resample — convert beforehand.
- **Tokenization** defaults to character-level, which is correct for Latin-script languages
  (German, English, …). Chinese/Japanese need pinyin/jieba segmentation — implement `IF5Tokenizer`
  and pass it via `F5TtsOptions.Tokenizer`.
- **NFE steps** (`F5TtsOptions.NfeSteps`, default 32) must match the value the transformer was
  exported with.

## Credits & license

- Library code: **MIT** (see [LICENSE](LICENSE)).
- Model architecture: [F5-TTS](https://github.com/SWivid/F5-TTS) (MIT).
- ONNX export tooling: [DakeQQ/F5-TTS-ONNX](https://github.com/DakeQQ/F5-TTS-ONNX) (Apache-2.0).
  The v0/Base-checkpoint fix that makes non-English fine-tunes export correctly was contributed
  upstream in [PR #74](https://github.com/DakeQQ/F5-TTS-ONNX/pull/74).
- The **model weights** you run carry their own license — e.g. the German checkpoint
  ([hvoss-techfak/F5-TTS-German](https://huggingface.co/hvoss-techfak/F5-TTS-German)) is
  **CC-BY-NC-4.0** (non-commercial).

## Support

This is free and MIT-licensed — no strings attached. If it saved you some time, you can
[buy me a coffee ☕](https://paypal.me/RobinLudwig240). Thanks!
