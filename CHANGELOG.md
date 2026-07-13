# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/nibor1896/Horus.F5Tts.Onnx/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/nibor1896/Horus.F5Tts.Onnx/releases/tag/v0.1.0
