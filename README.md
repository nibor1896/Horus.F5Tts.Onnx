# Horus.F5Tts.Onnx

[F5-TTS](https://github.com/SWivid/F5-TTS) inference for .NET on [ONNX Runtime](https://onnxruntime.ai/) — no Python, no PyTorch.
Input: a reference voice clip, its transcript, and the text to speak. Output: 24 kHz mono PCM.
Runs on CPU or on any GPU the ONNX Runtime build supports (DirectML for any DX12 GPU, CUDA for NVIDIA).

## Install

```sh
dotnet add package Horus.F5Tts.Onnx
```

The package references only the ONNX Runtime **managed** API. Add one native runtime package to select
where inference runs:

```sh
# CPU (works everywhere, slow for the ~1.3 GB transformer)
dotnet add package Microsoft.ML.OnnxRuntime

# any DirectX 12 GPU (NVIDIA / AMD / Intel)
dotnet add package Microsoft.ML.OnnxRuntime.DirectML

# NVIDIA CUDA
dotnet add package Microsoft.ML.OnnxRuntime.Gpu
```

## Models

Required files, from an F5-TTS ONNX export ([DakeQQ/F5-TTS-ONNX](https://github.com/DakeQQ/F5-TTS-ONNX)):

- `F5_Preprocess.onnx`
- `F5_Transformer.onnx`
- `F5_Decode.onnx`
- `vocab.txt`

Ready-made exports:

- German — [nibor1896/F5-TTS-German-ONNX](https://huggingface.co/nibor1896/F5-TTS-German-ONNX)
- English — [nibor1896/F5-TTS-English-ONNX](https://huggingface.co/nibor1896/F5-TTS-English-ONNX)

Other languages: export the checkpoint with DakeQQ's tooling. See [Languages & voices](#languages--voices).

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

// Reference audio at any sample rate; converted to the 24 kHz the model requires.
var referenceAudio = WavAudio.ReadPcm16Resampled("reference.wav", 24000);

var result = model.Synthesize(
    referenceAudio,
    referenceText: "This is the transcript of the reference clip.",
    text: "And this is the new sentence to speak.",
    new F5TtsOptions { Speed = 1.1f });

File.WriteAllBytes("out.wav", result.ToWav()); // 24 kHz mono WAV
```

## API

### Async

`Synthesize` is synchronous and CPU/GPU-bound. `SynthesizeAsync` runs it on a background thread and
takes a `CancellationToken`:

```csharp
var result = await model.SynthesizeAsync(
    referenceAudio, referenceText, text, cancellationToken: token);
```

Cancellation is checked **between denoising steps** — a call already inside ONNX Runtime is not
interrupted.

### PreparedVoice

Binds a reference once; later calls pass only the text:

```csharp
var voice = model.PrepareVoiceFromWav("reference.wav", referenceText);   // or PrepareVoice(short[], text)
var a = await voice.SynthesizeAsync("First line.");
var b = await voice.SynthesizeLongAsync(wholeParagraph);
```

No speed-up: each call runs the full pipeline, identical to passing the reference every time. F5 cannot
cache the reference across different texts, and that step is a fraction of a percent of the work.

### Progress

```csharp
var options = new F5TtsOptions
{
    Progress = new Progress<F5TtsProgress>(p => bar.Value = p.Fraction),
};
```

One report per denoising step. `Fraction` spans the whole request, including every chunk when
`SynthesizeLong` splits the text — 0→1 once, not per sentence. `Chunk` / `ChunkCount` give the
sentence index and count.

### Result

`Synthesize` returns an `F5TtsResult`:

| member | type | description |
|---|---|---|
| `Samples` | `short[]` | raw 16-bit PCM, **mono** |
| `SampleRate` | `int` | `24000` |
| `DurationSeconds` | `double` | length of the generated audio |
| `ToWav()` | `byte[]` | the samples encoded as an in-memory WAV file |

Write to file:

```csharp
File.WriteAllBytes("out.wav", result.ToWav());
```

Playback is not part of the library (no audio dependency). With
[NAudio](https://github.com/naudio/NAudio):

```csharp
using var ms = new MemoryStream(result.ToWav());
using var reader = new WaveFileReader(ms);
using var output = new WaveOutEvent();
output.Init(reader);
output.Play();
while (output.PlaybackState == PlaybackState.Playing) Thread.Sleep(100);
```

`result.Samples` is plain 24 kHz mono PCM for any other pipeline.

## Long text

One pass generates the reference clip **and** the new speech together; quality degrades once that
combined length passes ~22 seconds. The text budget therefore depends on the length of the reference clip.

`SynthesizeLong` computes the budget, splits at sentence boundaries into fitting pieces, synthesizes
each and cross-fades them:

```csharp
var result = model.SynthesizeLong(referenceAudio, referenceText, wholeParagraph);
// or: await model.SynthesizeLongAsync(referenceAudio, referenceText, wholeParagraph, cancellationToken: token);
```

Text that fits stays a single pass. `Seed` keeps the whole result reproducible: each chunk derives its
own seed from it, so the pieces get different noise and the output as a whole repeats exactly.

`TextChunker` is public for splitting the text yourself.

## Streaming

`SynthesizeLongAsync` returns after the last sentence. `SynthesizeStreamAsync` yields each chunk as it
is ready, so the first audio arrives after the first sentence:

```csharp
await foreach (var chunk in model.SynthesizeStreamAsync(referenceAudio, referenceText, wholeParagraph))
{
    // chunk.Samples is ready-to-play 24 kHz PCM — append it to your audio sink now.
    player.Write(chunk.Samples);          // e.g. sentence chunk.Index + 1 of chunk.Count
}
```

Concatenating every `chunk.Samples` in order gives **exactly** the same audio as `SynthesizeLongAsync`
for the same inputs and seed. Granularity is one chunk (F5-TTS generates each sentence's audio as a
whole); single-chunk text yields one item, equal to `SynthesizeAsync`. `CancellationToken` stops it
between or within chunks.

## Languages & voices

- **Voice** comes from the reference clip (plus its transcript) — the voice-cloning part.
- **Language and accent** come from the checkpoint. A German checkpoint speaks German; the base F5-TTS
  checkpoint speaks English and Chinese. The reference clip does not switch languages — English text on
  a German model produces garbled, wrongly-accented output.

Each checkpoint is one model set (three `.onnx` files + `vocab.txt`) and one `F5TtsModel`. For several
languages, load one model per language and route each request:

```csharp
using var german  = F5TtsModel.Load("models/de/F5_Preprocess.onnx", /* … */, "models/de/vocab.txt");
using var english = F5TtsModel.Load("models/en/F5_Preprocess.onnx", /* … */, "models/en/vocab.txt");

var result = german.Synthesize(refDe, referenceText: "Der Referenztext.", text: "Hallo, wie geht es dir?");
```

Loading is heavy — load once, keep the instance, do not reload per call.

Language-specific pieces:

- **Checkpoint and its `vocab.txt`.**
- **Tokenizer.** The default `CharTokenizer` (character-level) fits Latin-script languages — German,
  English, French, Spanish, …. Chinese/Japanese need pinyin/jieba segmentation: implement
  `IF5Tokenizer` and pass it via `F5TtsOptions.Tokenizer`.
- **Text normalizer** (optional, opt-in). `F5TtsOptions.TextNormalizer` spells out what the model would
  otherwise skip or mumble (`%`, `°C`, digits, `z.B.`, …); checkpoints are trained on normalised text,
  so raw numbers and symbols are out-of-distribution. Shipped: `GermanTextNormalizer`,
  `EnglishTextNormalizer`.

  ```csharp
  options.TextNormalizer = GermanTextNormalizer.Normalize;
  // "z.B. 50 % von 1.000 €"  -> "zum Beispiel fünfzig Prozent von eintausend Euro"
  // "am 3.8.2026 um 14:30 Uhr" -> "am dritten August zweitausendsechsundzwanzig um vierzehn Uhr dreißig"

  options.TextNormalizer = EnglishTextNormalizer.Normalize;
  // "I saved $1,000 (50%) by the 3rd, at 2:30 pm"
  //   -> "I saved one thousand dollars (fifty percent) by the third, at two thirty PM"
  ```

  Covered: numbers, percent, currency, decimals/thousands (German `,`/`.` convention), dates, clock
  times, article-governed ordinals (inflected from the leading word), abbreviations, a few symbols.
  Only recognised patterns are rewritten. Any other `Func<string, string>` works for other languages.

Pipeline, options and the 24 kHz audio format are identical across languages.

## Notes

- **Reference audio** must be 24 kHz mono. `WavAudio.ReadPcm16Resampled(path, 24000)` loads a 16-bit
  PCM WAV at any rate, down-mixes stereo and converts with a windowed-sinc kernel, so downsampling
  (44.1/48 kHz → 24 kHz) does not alias. `WavAudio.ReadPcm16` returns the file untouched.
- **NFE steps** (`F5TtsOptions.NfeSteps`, default 32) must match the value the transformer was exported
  with.
- **The reference clip's noise floor is inherited.** Measured with the stock F5-TTS demo clip:
  reference at −46.7 dBFS noise, output at −48 dBFS. Lowering the reference level does not help — the
  model normalises internally, so signal and noise rise together; a 3 dB quieter reference produced
  output at the same level with the same noise floor. What counts is the reference's SNR, not its
  volume.
- **The output can reach full scale and clip** — a few dozen samples in a 2.7 s clip in practice. This
  happens inside the decode graph, which emits `Int16` directly, so the peaks are already flattened
  before the library sees them; attenuating afterwards only makes the distortion quieter.
- **Half precision (FP16)** needs no different code and no extra setting — the library reads the
  precision off the model and marshals the right tensors. Match it to the execution provider (measured,
  same reference and text):

  | | F32 | FP16 |
  |---|---|---|
  | **GPU** (DirectML) | 617 ms / step | **60 ms / step** |
  | **CPU** | 19.6 s total | **40.1 s total** |

  Model size: 630 MB (FP16) vs 1.32 GB (F32). The CPU provider has no native half arithmetic, so ONNX
  Runtime emulates it and pays the conversions. **FP16 for GPU, F32 for CPU.**

  The same seed produces different audio on FP16 than on F32. Within one precision it reproduces
  exactly.

## Credits & license

- Library code: **MIT** (see [LICENSE](LICENSE)).
- Model architecture: [F5-TTS](https://github.com/SWivid/F5-TTS) (MIT).
- ONNX export tooling: [DakeQQ/F5-TTS-ONNX](https://github.com/DakeQQ/F5-TTS-ONNX) (Apache-2.0).
  The v0/Base-checkpoint fix for non-English fine-tunes was contributed upstream in
  [PR #74](https://github.com/DakeQQ/F5-TTS-ONNX/pull/74).
- **Model weights** carry their own license — e.g. the German checkpoint
  ([hvoss-techfak/F5-TTS-German](https://huggingface.co/hvoss-techfak/F5-TTS-German)) is
  **CC-BY-NC-4.0** (non-commercial).

## Support

[buy me a coffee ☕](https://paypal.me/RobinLudwig240)
