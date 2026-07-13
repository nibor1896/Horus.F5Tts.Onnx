namespace Horus.F5Tts.Onnx;

/// <summary>Per-synthesis options.</summary>
public sealed class F5TtsOptions
{
    /// <summary>Number of denoising steps (NFE). Higher = slightly better quality, linearly slower.
    /// Must match the value the transformer was exported with (default export: 32).</summary>
    public int NfeSteps { get; set; } = 32;

    /// <summary>Speaking-rate factor. Higher shortens the model's target duration for the same text,
    /// i.e. faster speech (not a time-stretch). 1.0 is the library default; ~1.1–1.2 often sounds
    /// more natural for assistant replies.</summary>
    public float Speed { get; set; } = 1.0f;

    /// <summary>Overrides the tokenizer. When null, the model's default character-level tokenizer is
    /// used (fine for Latin-script languages).</summary>
    public IF5Tokenizer? Tokenizer { get; set; }

    /// <summary>Optional text pre-processing applied to the generation text before tokenization
    /// (e.g. spelling out symbols like "%" or "°C" that the model would otherwise skip). Applied to
    /// the generation text only, not the reference text.</summary>
    public Func<string, string>? TextNormalizer { get; set; }

    /// <summary>Optional fixed seed for the initial diffusion noise. F5-TTS denoises from Gaussian
    /// noise; by default (null) that noise is drawn fresh each call, so the exact timbre varies
    /// slightly between runs. Set a seed to make synthesis <b>reproducible</b> — the same reference,
    /// text and seed always produce identical audio (useful for caching, tests, or a consistent
    /// assistant voice). Any value works; different seeds give different but equally stable results.
    /// The seeded generator (splitmix64) is platform-independent, so a seed reproduces across
    /// machines and execution providers.</summary>
    public int? Seed { get; set; }
}
