using Horus.F5Tts.Onnx;
using Xunit;

namespace Horus.F5Tts.Onnx.Tests;

public class ResampleTests
{
    private static short[] Sine(double frequencyHz, int sampleRate, int sampleCount, double amplitude = 10000)
    {
        var samples = new short[sampleCount];
        for (var i = 0; i < sampleCount; i++)
        {
            samples[i] = (short)Math.Round(amplitude * Math.Sin(2 * Math.PI * frequencyHz * i / sampleRate));
        }

        return samples;
    }

    private static double Rms(IReadOnlyList<short> samples)
    {
        double sum = 0;
        foreach (var s in samples)
        {
            sum += (double)s * s;
        }

        return Math.Sqrt(sum / samples.Count);
    }

    [Fact]
    public void Resample_returns_the_input_untouched_when_the_rates_match()
    {
        short[] samples = [1, 2, 3, -4];

        Assert.Equal(samples, WavAudio.Resample(samples, 24000, 24000));
    }

    [Fact]
    public void Resample_handles_an_empty_clip()
    {
        Assert.Empty(WavAudio.Resample([], 48000, 24000));
    }

    [Fact]
    public void Resample_halves_the_length_when_halving_the_rate()
    {
        var input = Sine(1000, 48000, 4800);

        var output = WavAudio.Resample(input, 48000, 24000);

        Assert.Equal(2400, output.Length);
    }

    [Fact]
    public void Resample_doubles_the_length_when_doubling_the_rate()
    {
        var input = Sine(1000, 24000, 2400);

        var output = WavAudio.Resample(input, 24000, 48000);

        Assert.Equal(4800, output.Length);
    }

    [Fact]
    public void Resample_handles_a_non_integer_ratio()
    {
        // The realistic case: a 44.1 kHz recording down to the model's 24 kHz.
        var input = Sine(1000, 44100, 44100);

        var output = WavAudio.Resample(input, 44100, 24000);

        Assert.Equal(24000, output.Length);
    }

    [Fact]
    public void Resample_preserves_a_constant_signal()
    {
        // A DC signal must come out at the same level — this is what catches a kernel whose
        // coefficients are not normalised (the audio would come out quieter or louder).
        var input = Enumerable.Repeat((short)1000, 2000).ToArray();

        var output = WavAudio.Resample(input, 48000, 24000);

        Assert.All(output, v => Assert.InRange(v, 998, 1002));
    }

    [Fact]
    public void Resample_reproduces_a_tone_well_below_the_cutoff()
    {
        // A 1 kHz tone is far below the 12 kHz Nyquist of the target rate, so resampling it from
        // 48 kHz must reproduce it almost exactly — compare against the same tone generated
        // directly at 24 kHz. Edges are skipped: the kernel clamps there by design.
        var input = Sine(1000, 48000, 9600);
        var expected = Sine(1000, 24000, 4800);

        var output = WavAudio.Resample(input, 48000, 24000);

        var error = 0.0;
        const int skip = 64;
        for (var i = skip; i < expected.Length - skip; i++)
        {
            error += Math.Abs(output[i] - expected[i]);
        }

        error /= expected.Length - 2 * skip;
        Assert.True(error < 500, $"mean absolute error {error:F1} is too high for a clean 1 kHz tone");
    }

    [Fact]
    public void Resample_suppresses_content_above_the_target_nyquist_instead_of_aliasing_it_back()
    {
        // The reason this is a windowed-sinc and not linear interpolation. A 20 kHz tone cannot
        // exist at 24 kHz (Nyquist 12 kHz). It must be filtered out — with linear interpolation it
        // would instead fold back into the audible range as a loud phantom tone, and the voice
        // cloned from such a clip would inherit it.
        var input = Sine(20000, 48000, 9600);

        var output = WavAudio.Resample(input, 48000, 24000);

        var ratio = Rms(output) / Rms(input);
        Assert.True(ratio < 0.2, $"20 kHz survived resampling at {ratio:P0} of its level — aliasing");
    }

    [Theory]
    [InlineData(0, 24000)]
    [InlineData(-1, 24000)]
    [InlineData(48000, 0)]
    [InlineData(48000, -1)]
    public void Resample_rejects_non_positive_sample_rates(int source, int target)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => WavAudio.Resample([1, 2, 3], source, target));
    }

    [Fact]
    public void ReadPcm16Resampled_loads_and_converts_in_one_step()
    {
        var wav = WavAudio.WritePcm16(Sine(1000, 48000, 4800), 48000);

        var samples = WavAudio.ReadPcm16Resampled(new MemoryStream(wav), 24000);

        Assert.Equal(2400, samples.Length);
    }

    [Fact]
    public void ReadPcm16Resampled_leaves_a_clip_that_is_already_at_the_target_rate_alone()
    {
        var original = Sine(1000, 24000, 2400);
        var wav = WavAudio.WritePcm16(original, 24000);

        var samples = WavAudio.ReadPcm16Resampled(new MemoryStream(wav), 24000);

        Assert.Equal(original, samples);
    }
}
