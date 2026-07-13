namespace Horus.F5Tts.Onnx;

/// <summary>Minimal, dependency-free RIFF/WAV helpers for 16-bit PCM. Just enough to load a
/// reference clip and to save the synthesized result — no resampling. F5-TTS works at 24 kHz mono;
/// convert your reference clips to that beforehand (any audio tool does this).</summary>
public static class WavAudio
{
    /// <summary>Reads a 16-bit PCM WAV file into mono samples. Stereo input is down-mixed. The
    /// original sample rate is returned unchanged (no resampling) — the caller is responsible for
    /// ensuring it matches what the model expects (24 kHz).</summary>
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
