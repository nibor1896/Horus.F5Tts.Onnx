# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `F5TtsOptions.Seed` (optional) — a fixed seed for the initial diffusion noise. F5-TTS denoises
  from Gaussian noise drawn fresh each call, so timbre varies slightly between runs; setting a seed
  makes synthesis reproducible (same reference + text + seed → identical audio). Uses a
  platform-independent splitmix64 generator, so a seed reproduces across machines and execution
  providers. Left `null` by default, preserving the natural per-call variation. (Mirrors the fix the
  Horus app uses to keep its assistant voice consistent — `System.Random` was deliberately avoided
  because some of its draws destabilized the denoiser tail on DirectML.)

## [0.1.1] - 2026-07-13

Re-release of 0.1.0. The 0.1.0 package never became available on nuget.org — after an initial
manual push its validation stalled, which permanently reserves the version number (every later
push then returns "409 already exists" and publishes nothing). 0.1.1 is a fresh version so the
release workflow can publish cleanly. **No API or behavior changes since 0.1.0.**

### Added
- Project landing page (`docs/index.html`, served via GitHub Pages) — minimalist, with a
  light/dark theme toggle.

### Changed
- README now leads with the "first pure-.NET F5-TTS runner" framing.
- Landing page: clarified that the library is language-agnostic (a German model is just the
  ready-to-use example), and corrected the footer license note to reference the German model
  specifically rather than "all models".
- README links the DEV.to write-up ("Read the story").
- README documents the `F5TtsResult` (what `Synthesize` returns) with save-to-file and playback
  examples.
- Landing page footer links the README/docs (under "Support").

## [0.1.0] - 2026-07-13

Initial release.

### Added
- `F5TtsModel` — loads the three-model F5-TTS ONNX export (Preprocess / Transformer / Decode) plus
  its `vocab.txt`, and synthesizes speech from a reference voice clip + text entirely through ONNX
  Runtime (no Python). Runs the Preprocess → NFE denoising loop → Decode pipeline in-process and
  serializes concurrent calls.
- `CharTokenizer` (default) with a pluggable `IF5Tokenizer` interface — character-level tokenization
  is correct for Latin-script languages; the interface is the extension point for CJK.
- `F5TtsOptions` — NFE steps, speaking speed, custom tokenizer, and an optional text normalizer.
- Consumer-selectable execution provider (CPU / DirectML / CUDA) via a `configureSession` hook, so
  the library never forces a native runtime on its users (it depends only on the ONNX Runtime
  managed API).
- `WavAudio` — dependency-free 16-bit PCM WAV read/write helpers.
- Console sample (`samples/`) and GitHub Actions workflows: CI (build + pack) and NuGet publish on
  release via NuGet Trusted Publishing (OIDC — no API key or secret to manage).

### Fixed (pre-release incidents)
- **Source tree excluded from the first commit.** The `.gitignore` rule `*.onnx` also matched the
  `Horus.F5Tts.Onnx` project *directory* (Git is case-insensitive on Windows, `.Onnx` ≈ `*.onnx`),
  so the entire `src/` folder was silently ignored and the first CI run failed with "project file
  not found". Fixed by scoping model-file ignores to a `models/` folder instead of a bare
  `*.onnx`.
- **`DenseTensor<T>` overload ambiguity.** Constructing tensors with collection-expression
  arguments (`new DenseTensor<long>([x], [1])`) bound to the wrong constructor overload
  (CS0029 / CS9174). Switched to explicit arrays. Caught by CI before any release.
- **NFE loop ran one step too many.** The denoising loop iterated `NFE` times feeding a fresh step
  index, overrunning the transformer's time-step table (`Gather` out-of-bounds at index 31). The
  driver actually performs `NFE - 1` steps, starting `time_step` at 0 and feeding back the value the
  transformer returns each iteration. Fixed and verified end-to-end (Whisper large-v3: de, 1.00).

[Unreleased]: https://github.com/nibor1896/Horus.F5Tts.Onnx/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/nibor1896/Horus.F5Tts.Onnx/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/nibor1896/Horus.F5Tts.Onnx/releases/tag/v0.1.0
