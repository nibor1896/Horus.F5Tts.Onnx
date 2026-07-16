using Horus.F5Tts.Onnx;
using Xunit;

namespace Horus.F5Tts.Onnx.Tests;

public class F5TtsResultTests
{
    [Fact]
    public void DurationSeconds_is_sample_count_over_sample_rate()
    {
        var result = new F5TtsResult(new short[24000], 24000);

        Assert.Equal(1.0, result.DurationSeconds, 6);
    }

    [Fact]
    public void DurationSeconds_is_zero_for_an_invalid_sample_rate()
    {
        var result = new F5TtsResult(new short[100], 0);

        Assert.Equal(0, result.DurationSeconds);
    }

    [Fact]
    public void ToWav_produces_audio_that_reads_back_unchanged()
    {
        short[] samples = [5, -5, 1000, -1000];
        var result = new F5TtsResult(samples, 24000);

        var (read, rate) = WavAudio.ReadPcm16(new MemoryStream(result.ToWav()));

        Assert.Equal(samples, read);
        Assert.Equal(24000, rate);
    }
}
