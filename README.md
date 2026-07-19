> 🚧 **On `main`, not yet on NuGet:** streaming synthesis (`SynthesizeStreamAsync`), German & English
> text normalizers (`GermanTextNormalizer` / `EnglishTextNormalizer`), `PreparedVoice`, and progress
> reporting via `IProgress<T>`.
> The published package is **0.2.0** — see the [CHANGELOG](https://github.com/nibor1896/Horus.F5Tts.Onnx/blob/main/CHANGELOG.md#unreleased) for what has landed since.

# Horus.F5Tts.Onnx

> ### The first pure-.NET runner for [F5-TTS](https://github.com/SWivid/F5-TTS).
> Until now, running F5-TTS meant Python. This library runs it entirely on
> [ONNX Runtime](https://onnxruntime.ai/) — **no Python, no PyTorch** — from any .NET app.

Give it a short reference voice clip and some text, get 24 kHz audio back. Runs on CPU or any GPU
your ONNX Runtime build supports (DirectML for any DX12 GPU, CUDA for NVIDIA).

📖 **Read the story:** [Shipping the first .NET F5-TTS library — and the ONNX bug I had to fix first](https://dev.to/nibor1896/shipping-the-first-net-f5-tts-library-and-the-onnx-bug-i-had-to-fix-first-22dc)

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

Ready-made exports:

- **German** — [nibor1896/F5-TTS-German-ONNX](https://huggingface.co/nibor1896/F5-TTS-German-ONNX)
- **English** — [nibor1896/F5-TTS-English-ONNX](https://huggingface.co/nibor1896/F5-TTS-English-ONNX)

For other languages, export the checkpoint yourself with DakeQQ's tooling. See
[Languages & voices](#languages--voices) below for how the pieces fit together.

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

// Reference voice + its transcript. Any sample rate: this converts it to the 24 kHz the model wants.
var referenceAudio = WavAudio.ReadPcm16Resampled("reference.wav", 24000);

var result = model.Synthesize(
    referenceAudio,
    referenceText: "This is the transcript of the reference clip.",
    text: "And this is the new sentence to speak.",
    new F5TtsOptions { Speed = 1.1f });

File.WriteAllBytes("out.wav", result.ToWav()); // 24 kHz mono WAV
```

`Synthesize` is synchronous and CPU/GPU-bound. In a UI or server app use `SynthesizeAsync`, which
runs it on a background thread and takes a `CancellationToken`:

```csharp
var result = await model.SynthesizeAsync(
    referenceAudio, referenceText, text, cancellationToken: token);
```

Cancellation is honoured **between denoising steps**, so a long request can be abandoned part-way
instead of only before it starts (a step is the granularity — a call already inside ONNX Runtime
can't be interrupted).

When one voice speaks many things, bind the reference once with `PrepareVoice` and then pass only the
text:

```csharp
var voice = model.PrepareVoiceFromWav("reference.wav", referenceText);   // or PrepareVoice(short[], text)
var a = await voice.SynthesizeAsync("First line.");
var b = await voice.SynthesizeLongAsync(wholeParagraph);
```

It is a convenience, not a speed-up — each call runs the full pipeline, identical to passing the
reference every time (F5 can't cache the reference across different texts, and that step is a
fraction of a percent of the work anyway).

Synthesis is slow and silent, so for a progress bar set `F5TtsOptions.Progress`:

```csharp
var options = new F5TtsOptions
{
    Progress = new Progress<F5TtsProgress>(p => bar.Value = p.Fraction),
};
```

You get a report after every denoising step. `Fraction` spans the **whole request** — including every
chunk when `SynthesizeLong` splits the text — so the bar runs 0→1 once instead of restarting at each
sentence. `Chunk` / `ChunkCount` are there when you want to write "sentence 3 of 7" next to it.

## What you get back

`Synthesize` returns an `F5TtsResult`:

| member | type | description |
|---|---|---|
| `Samples` | `short[]` | raw 16-bit PCM, **mono** |
| `SampleRate` | `int` | `24000` |
| `DurationSeconds` | `double` | length of the generated audio |
| `ToWav()` | `byte[]` | the samples encoded as an in-memory WAV file |

**Save it** to a file:

```csharp
File.WriteAllBytes("out.wav", result.ToWav());
```

**Play it** — the library has no audio output of its own (to stay dependency-light), so use any
player. With [NAudio](https://github.com/naudio/NAudio):

```csharp
using var ms = new MemoryStream(result.ToWav());
using var reader = new WaveFileReader(ms);
using var output = new WaveOutEvent();
output.Init(reader);
output.Play();
while (output.PlaybackState == PlaybackState.Playing) Thread.Sleep(100);
```

Or feed `result.Samples` straight into your own audio pipeline — it's plain 24 kHz mono PCM.

## Long text

A single pass generates the reference clip **and** the new speech together, and quality falls apart
once that combined length runs much past ~22 seconds. The usable text budget is therefore not a fixed
number — it depends on how much of the pass your reference clip already eats.

`SynthesizeLong` deals with that: it works the budget out, splits at sentence boundaries into pieces
that fit, synthesizes each and cross-fades them together.

```csharp
var result = model.SynthesizeLong(referenceAudio, referenceText, wholeParagraph);
// or: await model.SynthesizeLongAsync(referenceAudio, referenceText, wholeParagraph, cancellationToken: token);
```

Text that already fits stays a single pass, so there is nothing to lose by reaching for it by default.
A `Seed` still makes the whole result reproducible: each chunk derives its own seed from it, so the
pieces get different noise the way they would inside one pass, and the output as a whole repeats
exactly.

`TextChunker` is public if you would rather split the text yourself.

## Streaming (speak sooner)

For long text, `SynthesizeLongAsync` returns nothing until every sentence is done — so the user waits
for the whole paragraph before hearing a word. `SynthesizeStreamAsync` yields each piece as it is
ready, so the **first audio arrives after the first sentence**:

```csharp
await foreach (var chunk in model.SynthesizeStreamAsync(referenceAudio, referenceText, wholeParagraph))
{
    // chunk.Samples is ready-to-play 24 kHz PCM — append it to your audio sink now.
    player.Write(chunk.Samples);          // e.g. sentence chunk.Index + 1 of chunk.Count
}
```

Concatenating every `chunk.Samples` in order gives **exactly** the same audio as
`SynthesizeLongAsync` for the same inputs and seed — the stream is the batch result delivered
incrementally, not a different rendering. It is chunk-granularity streaming (F5-TTS generates each
sentence's audio as a whole), so the gain is the first chunk arriving early; short single-chunk text
yields one item, the same as `SynthesizeAsync`. `CancellationToken` stops it promptly, between or
within chunks.

## Languages & voices

Two independent things decide how the output sounds:

- **The speaking voice comes from the reference clip.** Whoever you pass as the reference audio
  (plus its transcript) is the voice you get back — that's the voice-cloning part.
- **The language and accent come from the checkpoint.** A German checkpoint speaks German; the base
  F5-TTS checkpoint speaks English (and Chinese). The reference clip does **not** switch languages —
  feed English text to a German model and you get garbled, wrongly-accented output.

So: pick the **checkpoint** for the language, pick the **reference clip** for the voice.

### Using a different-language checkpoint

Each checkpoint is one **model set** (three `.onnx` files + `vocab.txt`) and one `F5TtsModel`. To
support several languages, load one model per language and route each request to the matching one:

```csharp
using var german  = F5TtsModel.Load("models/de/F5_Preprocess.onnx", /* … */, "models/de/vocab.txt");
using var english = F5TtsModel.Load("models/en/F5_Preprocess.onnx", /* … */, "models/en/vocab.txt");

var result = german.Synthesize(refDe, referenceText: "Der Referenztext.", text: "Hallo, wie geht es dir?");
```

Loading is heavy — load the models you need once and keep them; don't reload per call.

The only language-specific pieces are:

- **The checkpoint and its `vocab.txt`** — the model itself.
- **The tokenizer.** The default `CharTokenizer` (character-level) is correct for Latin-script
  languages — German, English, French, Spanish, …. Chinese/Japanese need pinyin/jieba segmentation:
  implement `IF5Tokenizer` and pass it via `F5TtsOptions.Tokenizer`.
- **The text normalizer** (optional). `F5TtsOptions.TextNormalizer` spells out what the model would
  otherwise skip or mumble (`%`, `°C`, digits, `z.B.`, …) — checkpoints are trained on normalised
  text, so raw numbers and symbols are out-of-distribution. What to spell out is language-specific, so
  it is opt-in; the library ships ready defaults for German (`GermanTextNormalizer`) and English
  (`EnglishTextNormalizer`):

  ```csharp
  options.TextNormalizer = GermanTextNormalizer.Normalize;
  // "z.B. 50 % von 1.000 €"  -> "zum Beispiel fünfzig Prozent von eintausend Euro"
  // "am 3.8.2026 um 14:30 Uhr" -> "am dritten August zweitausendsechsundzwanzig um vierzehn Uhr dreißig"

  options.TextNormalizer = EnglishTextNormalizer.Normalize;
  // "I saved $1,000 (50%) by the 3rd, at 2:30 pm"
  //   -> "I saved one thousand dollars (fifty percent) by the third, at two thirty PM"
  ```

  It handles numbers, percent, currency, decimals/thousands (German `,`/`.` convention), dates, clock
  times, article-governed ordinals (inflected from the leading word), abbreviations and a few symbols,
  rewriting only recognised patterns and leaving prose untouched. Supply any other
  `Func<string, string>` for a different language.

Everything else — the pipeline, the options, the 24 kHz audio format — is identical across languages.

## Notes

- **Reference audio** must end up 24 kHz mono. `WavAudio.ReadPcm16Resampled(path, 24000)` loads a
  16-bit PCM WAV at *any* rate, down-mixes stereo and converts it for you — with a windowed-sinc
  kernel, so downsampling (44.1/48 kHz → 24 kHz) does not alias. `WavAudio.ReadPcm16` still returns
  the file untouched if you'd rather handle the rate yourself.
- **NFE steps** (`F5TtsOptions.NfeSteps`, default 32) must match the value the transformer was
  exported with.
- **The reference clip's noise is inherited — use a clean recording, not a quiet one.** Voice cloning
  copies the voice *and* its noise floor. Measured with the stock F5-TTS demo clip: the reference sits
  at −46.7 dBFS of noise and the output lands at −48 dBFS, i.e. right behind it. Turning the reference
  down does **not** help: the model normalises it internally, so signal and noise come back up
  together. A 3 dB quieter reference measurably produced output at the *same* level with the *same*
  noise floor. What matters is the reference's signal-to-noise ratio, not its volume.
- **The output can reach full scale and clip a little** — a few dozen samples in a 2.7 s clip, in
  practice. This happens inside the decode graph, which emits `Int16` directly, so the peaks are
  already flattened before the library ever sees them; attenuating afterwards would only make the
  distortion quieter. It is a property of the model, not something this library can undo.
- **Half precision (FP16)** works out of the box — the library reads the precision off the model and
  marshals the right tensors, so an FP16 export needs no different code and no extra setting. But
  **match it to your execution provider**, because it cuts both ways (measured, same reference and
  text):

  | | F32 | FP16 |
  |---|---|---|
  | **GPU** (DirectML) | 617 ms / step | **60 ms / step** |
  | **CPU** | 19.6 s total | **40.1 s total** |

  On a GPU it is the single biggest win available, and the model is half the size (630 MB vs
  1.32 GB). On the CPU provider it is a **loss**: there is no native half arithmetic there, so ONNX
  Runtime emulates it and pays the conversions for nothing. **FP16 for GPU, F32 for CPU.**

  The same seed produces *different* audio on FP16 than on F32 — fewer bits, different numbers.
  Within one precision it reproduces exactly, as documented.

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
