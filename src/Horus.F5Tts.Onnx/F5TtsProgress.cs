namespace Horus.F5Tts.Onnx;

/// <summary>How far along a synthesis is. Reported to <see cref="F5TtsOptions.Progress"/>.
///
/// Carries both the raw counters and a ready-made <see cref="Fraction"/>: a progress bar wants one
/// number, but a label usually wants to say "sentence 3 of 7", and throwing that away to save a field
/// would be a false economy.</summary>
/// <param name="Chunk">Zero-based index of the chunk being spoken. Always 0 for a single pass.</param>
/// <param name="ChunkCount">How many chunks the text was split into. 1 for a single pass.</param>
/// <param name="Step">Denoising steps completed within the current chunk.</param>
/// <param name="StepCount">Denoising steps this chunk will take.</param>
public readonly record struct F5TtsProgress(int Chunk, int ChunkCount, int Step, int StepCount)
{
    /// <summary>Progress across the whole request, 0 to 1 — chunks included, so it does not restart at
    /// each sentence. Reaches exactly 1 when the last chunk's last step is done; the decode that
    /// follows takes about a second and is not counted, because reporting 99 % during it would be
    /// noisier than useful.</summary>
    public double Fraction => ChunkCount <= 0 || StepCount <= 0
        ? 0
        : Math.Clamp((Chunk + (double)Step / StepCount) / ChunkCount, 0, 1);
}
