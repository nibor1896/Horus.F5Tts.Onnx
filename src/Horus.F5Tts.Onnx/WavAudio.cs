namespace Horus.F5Tts.Onnx;

/// <summary>Minimal, dependency-free RIFF/WAV helpers for 16-bit PCM. Just enough to load a
/// reference clip and to save the synthesized result. F5-TTS works at 24 kHz mono:
/// <see cref="ReadPcm16Resampled(string, int)"/> loads a clip at any rate and converts it for you,
/// while <see cref="ReadPcm16(string)"/> hands back exactly what is in the file.</summary>
public static class WavAudio
{
    /// <summary>Lanczos lobes for <see cref="Resample"/>. Eight is a good quality/cost trade for
    /// speech; the kernel widens automatically when downsampling.</summary>
    private const int ResampleLobes = 8;

    /// <summary>Reads a 16-bit PCM WAV file into mono samples. Stereo input is down-mixed. The
    /// original sample rate is returned unchanged — this overload does <b>not</b> resample. Use
    /// <see cref="ReadPcm16Resampled(string, int)"/> when the clip may not already be at the rate
    /// the model expects (24 kHz).</summary>
    public static (short[] Samples, int SampleRate) ReadPcm16(string path)
    {
        using var stream = File.OpenRead(path);
        return ReadPcm16(stream);
    }

    /// <summary>Reads a 16-bit PCM WAV stream into mono samples (see <see cref="ReadPcm16(string)"/>).</summary>
    public static (short[] Samples, int SampleRate) ReadPcm16(Stream stream)
    {
        using var reader = new BinaryReader(stream);

        if (new string(reader.ReadChars(4)) != "RIFF")
        {
            throw new InvalidDataException("Not a RIFF/WAV file.");
        }

        reader.ReadUInt32(); // overall size
        if (new string(reader.ReadChars(4)) != "WAVE")
        {
            throw new InvalidDataException("Not a WAVE file.");
        }

        int channels = 1, sampleRate = 24000, bitsPerSample = 16;
        short[]? samples = null;

        while (stream.Position < stream.Length)
        {
            var chunkId = new string(reader.ReadChars(4));
            var chunkSize = reader.ReadUInt32();

            if (chunkId == "fmt ")
            {
                reader.ReadUInt16();                 // audio format (1 = PCM)
                channels = reader.ReadUInt16();
                sampleRate = (int)reader.ReadUInt32();
                reader.ReadUInt32();                 // byte rate
                reader.ReadUInt16();                 // block align
                bitsPerSample = reader.ReadUInt16();
                var remaining = (int)chunkSize - 16; // skip any extended fmt bytes
                if (remaining > 0)
                {
                    reader.ReadBytes(remaining);
                }
            }
            else if (chunkId == "data")
            {
                if (bitsPerSample != 16)
                {
                    throw new NotSupportedException($"Only 16-bit PCM is supported (got {bitsPerSample}-bit).");
                }

                var bytes = reader.ReadBytes((int)chunkSize);
                var total = bytes.Length / 2;
                var interleaved = new short[total];
                Buffer.BlockCopy(bytes, 0, interleaved, 0, total * 2);

                if (channels <= 1)
                {
                    samples = interleaved;
                }
                else
                {
                    // Down-mix to mono by averaging channels.
                    var frames = total / channels;
                    samples = new short[frames];
                    for (var f = 0; f < frames; f++)
                    {
                        var sum = 0;
                        for (var c = 0; c < channels; c++)
                        {
                            sum += interleaved[f * channels + c];
                        }

                        samples[f] = (short)(sum / channels);
                    }
                }
            }
            else
            {
                // Unknown chunk - skip it (chunks are word-aligned).
                reader.ReadBytes((int)chunkSize + ((int)chunkSize & 1));
            }
        }

        if (samples is null)
        {
            throw new InvalidDataException("WAV file has no data chunk.");
        }

        return (samples, sampleRate);
    }

    /// <summary>Reads a 16-bit PCM WAV file as mono at <paramref name="targetSampleRate"/>, resampling
    /// if the file uses a different rate (stereo is down-mixed first). This is the one-liner for
    /// loading a reference clip that is not already at the model's 24 kHz.</summary>
    /// <param name="path">Path to a 16-bit PCM WAV file, at any sample rate.</param>
    /// <param name="targetSampleRate">The rate to convert to, e.g. <c>24000</c> for F5-TTS.</param>
    public static short[] ReadPcm16Resampled(string path, int targetSampleRate)
    {
        using var stream = File.OpenRead(path);
        return ReadPcm16Resampled(stream, targetSampleRate);
    }

    /// <summary>Reads a 16-bit PCM WAV stream as mono at <paramref name="targetSampleRate"/>
    /// (see <see cref="ReadPcm16Resampled(string, int)"/>).</summary>
    public static short[] ReadPcm16Resampled(Stream stream, int targetSampleRate)
    {
        var (samples, sampleRate) = ReadPcm16(stream);
        return Resample(samples, sampleRate, targetSampleRate);
    }

    /// <summary>Resamples mono 16-bit PCM from one rate to another using a windowed-sinc (Lanczos)
    /// kernel. Returns the input unchanged when the rates already match.
    ///
    /// Deliberately not linear interpolation: the common case is <i>downsampling</i> (48/44.1 kHz
    /// down to the model's 24 kHz), where linear interpolation folds the discarded high frequencies
    /// back into the audible range as aliasing. The reference clip is what the voice is cloned from,
    /// so that distortion would be inherited by every synthesized sentence. The kernel here doubles
    /// as the anti-alias low-pass.</summary>
    /// <param name="samples">Mono 16-bit PCM samples.</param>
    /// <param name="sourceSampleRate">The rate <paramref name="samples"/> is currently at.</param>
    /// <param name="targetSampleRate">The rate to convert to.</param>
    /// <exception cref="ArgumentOutOfRangeException">A sample rate is not positive.</exception>
    public static short[] Resample(ReadOnlySpan<short> samples, int sourceSampleRate, int targetSampleRate)
    {
        if (sourceSampleRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sourceSampleRate), sourceSampleRate,
                "Sample rate must be positive.");
        }

        if (targetSampleRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(targetSampleRate), targetSampleRate,
                "Sample rate must be positive.");
        }

        if (sourceSampleRate == targetSampleRate || samples.Length == 0)
        {
            return samples.ToArray();
        }

        var ratio = (double)targetSampleRate / sourceSampleRate;
        var outLength = Math.Max(1, (int)Math.Round(samples.Length * ratio));
        var result = new short[outLength];

        // Downsampling drops the cutoff to the target's Nyquist; upsampling keeps the source's.
        var cutoff = Math.Min(1.0, ratio);
        var halfWidth = ResampleLobes / cutoff;

        for (var i = 0; i < outLength; i++)
        {
            var center = i / ratio;
            var first = (int)Math.Ceiling(center - halfWidth);
            var last = (int)Math.Floor(center + halfWidth);

            double sum = 0, norm = 0;
            for (var n = first; n <= last; n++)
            {
                var x = center - n;
                var coeff = Sinc(cutoff * x) * Sinc(x / halfWidth); // sinc kernel * Lanczos window

                // Clamp at the edges: repeating the first/last sample beats fading into silence.
                var index = n < 0 ? 0 : n >= samples.Length ? samples.Length - 1 : n;
                sum += samples[index] * coeff;
                norm += coeff;
            }

            // Normalising by the kernel sum keeps the gain correct, including at the clamped edges.
            var value = norm != 0 ? sum / norm : 0;
            result[i] = (short)Math.Clamp(Math.Round(value), short.MinValue, short.MaxValue);
        }

        return result;
    }

    /// <summary>Normalised sinc: sin(pi*x) / (pi*x), and 1 at x = 0.</summary>
    private static double Sinc(double x)
    {
        if (Math.Abs(x) < 1e-9)
        {
            return 1.0;
        }

        var piX = Math.PI * x;
        return Math.Sin(piX) / piX;
    }

    /// <summary>Encodes 16-bit PCM mono samples as an in-memory WAV file.</summary>
    public static byte[] WritePcm16(ReadOnlySpan<short> samples, int sampleRate)
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);
        var dataLen = samples.Length * 2;

        bw.Write("RIFF"u8.ToArray());
        bw.Write(36 + dataLen);
        bw.Write("WAVE"u8.ToArray());
        bw.Write("fmt "u8.ToArray());
        bw.Write(16);                  // PCM fmt chunk size
        bw.Write((short)1);            // PCM
        bw.Write((short)1);            // mono
        bw.Write(sampleRate);
        bw.Write(sampleRate * 2);      // byte rate (16-bit mono)
        bw.Write((short)2);            // block align
        bw.Write((short)16);           // bits per sample
        bw.Write("data"u8.ToArray());
        bw.Write(dataLen);
        foreach (var s in samples)
        {
            bw.Write(s);
        }

        bw.Flush();
        return ms.ToArray();
    }
}
