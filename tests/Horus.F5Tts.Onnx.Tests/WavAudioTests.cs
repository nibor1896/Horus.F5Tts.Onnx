using Horus.F5Tts.Onnx;
using Xunit;

namespace Horus.F5Tts.Onnx.Tests;

public class WavAudioTests
{
    /// <summary>Builds a RIFF/WAV byte blob by hand, so the tests exercise the parser against
    /// files it did not write itself.</summary>
    private static byte[] BuildWav(int channels, int sampleRate, short bitsPerSample, byte[] data,
        byte[]? unknownChunk = null)
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);
        var blockAlign = (short)(channels * bitsPerSample / 8);

        bw.Write("RIFF"u8.ToArray());
        bw.Write(0); // size field: the reader skips it
        bw.Write("WAVE"u8.ToArray());

        bw.Write("fmt "u8.ToArray());
        bw.Write(16);
        bw.Write((short)1); // PCM
        bw.Write((short)channels);
        bw.Write(sampleRate);
        bw.Write(sampleRate * blockAlign);
        bw.Write(blockAlign);
        bw.Write(bitsPerSample);

        if (unknownChunk is not null)
        {
            // Even length keeps the chunk word-aligned, as the format requires.
            bw.Write("LIST"u8.ToArray());
            bw.Write(unknownChunk.Length);
            bw.Write(unknownChunk);
        }

        bw.Write("data"u8.ToArray());
        bw.Write(data.Length);
        bw.Write(data);
        bw.Flush();
        return ms.ToArray();
    }

    private static byte[] ToBytes(params short[] samples)
    {
        var bytes = new byte[samples.Length * 2];
        Buffer.BlockCopy(samples, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    [Fact]
    public void WritePcm16_then_ReadPcm16_round_trips_samples_and_rate()
    {
        short[] samples = [0, 1, -1, short.MaxValue, short.MinValue, 1234];

        var wav = WavAudio.WritePcm16(samples, 24000);
        var (read, rate) = WavAudio.ReadPcm16(new MemoryStream(wav));

        Assert.Equal(samples, read);
        Assert.Equal(24000, rate);
    }

    [Fact]
    public void ReadPcm16_down_mixes_stereo_by_averaging_channels()
    {
        // Two frames: (100, 200) -> 150, (-10, -20) -> -15.
        var wav = BuildWav(channels: 2, sampleRate: 24000, bitsPerSample: 16,
            data: ToBytes(100, 200, -10, -20));

        var (samples, _) = WavAudio.ReadPcm16(new MemoryStream(wav));

        Assert.Equal(new short[] { 150, -15 }, samples);
    }

    [Fact]
    public void ReadPcm16_returns_the_original_rate_and_does_not_resample()
    {
        // The library deliberately does not resample — a 44.1 kHz file comes back at 44.1 kHz with
        // its sample count untouched, and it is the caller's job to convert.
        var wav = BuildWav(channels: 1, sampleRate: 44100, bitsPerSample: 16, data: ToBytes(1, 2, 3, 4));

        var (samples, rate) = WavAudio.ReadPcm16(new MemoryStream(wav));

        Assert.Equal(44100, rate);
        Assert.Equal(4, samples.Length);
    }

    [Fact]
    public void ReadPcm16_skips_unknown_chunks()
    {
        var wav = BuildWav(channels: 1, sampleRate: 24000, bitsPerSample: 16,
            data: ToBytes(7, 8), unknownChunk: [1, 2, 3, 4]);

        var (samples, _) = WavAudio.ReadPcm16(new MemoryStream(wav));

        Assert.Equal(new short[] { 7, 8 }, samples);
    }

    [Fact]
    public void ReadPcm16_rejects_a_non_riff_file()
    {
        var junk = new byte[] { 0x4A, 0x55, 0x4E, 0x4B, 0, 0, 0, 0, 0x57, 0x41, 0x56, 0x45 };

        Assert.Throws<InvalidDataException>(() => WavAudio.ReadPcm16(new MemoryStream(junk)));
    }

    [Fact]
    public void ReadPcm16_rejects_non_16_bit_audio()
    {
        var wav = BuildWav(channels: 1, sampleRate: 24000, bitsPerSample: 8, data: [1, 2, 3, 4]);

        Assert.Throws<NotSupportedException>(() => WavAudio.ReadPcm16(new MemoryStream(wav)));
    }

    [Fact]
    public void WritePcm16_emits_a_mono_pcm16_header()
    {
        var wav = WavAudio.WritePcm16([1, 2, 3], 24000);

        Assert.Equal("RIFF", System.Text.Encoding.ASCII.GetString(wav, 0, 4));
        Assert.Equal("WAVE", System.Text.Encoding.ASCII.GetString(wav, 8, 4));
        Assert.Equal("fmt ", System.Text.Encoding.ASCII.GetString(wav, 12, 4));
        Assert.Equal(1, BitConverter.ToInt16(wav, 20));      // PCM
        Assert.Equal(1, BitConverter.ToInt16(wav, 22));      // mono
        Assert.Equal(24000, BitConverter.ToInt32(wav, 24));  // sample rate
        Assert.Equal(16, BitConverter.ToInt16(wav, 34));     // bits per sample
        Assert.Equal("data", System.Text.Encoding.ASCII.GetString(wav, 36, 4));
        Assert.Equal(6, BitConverter.ToInt32(wav, 40));      // 3 samples * 2 bytes
        Assert.Equal(44 + 6, wav.Length);
    }

    [Fact]
    public void WritePcm16_handles_an_empty_clip()
    {
        var wav = WavAudio.WritePcm16([], 24000);
        var (samples, rate) = WavAudio.ReadPcm16(new MemoryStream(wav));

        Assert.Empty(samples);
        Assert.Equal(24000, rate);
    }
}
