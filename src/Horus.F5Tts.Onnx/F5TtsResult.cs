namespace Horus.F5Tts.Onnx;

/// <summary>The synthesized audio: 16-bit PCM samples plus their sample rate (F5-TTS outputs 24 kHz
/// mono). Use <see cref="WavAudio.WritePcm16"/> to turn it into a playable .wav.</summary>
public sealed class F5TtsResult
{
    internal F5TtsResult(short[] samples, int sampleRate)
    {
        Samples = samples;
        SampleRate = sampleRate;
    }

    /// <summary>Signed 16-bit PCM, mono.</summary>
    public short[] Samples { get; }

    /// <summary>Samples per second (24000 for F5-TTS).</summary>
    public int SampleRate { get; }

    /// <summary>Length of the generated audio in seconds.</summary>
    public double DurationSeconds => SampleRate > 0 ? (double)Samples.Length / SampleRate : 0;

    /// <summary>Encodes the samples as an in-memory WAV file (RIFF/PCM16).</summary>
    public byte[] ToWav() => WavAudio.WritePcm16(Samples, SampleRate);
}
