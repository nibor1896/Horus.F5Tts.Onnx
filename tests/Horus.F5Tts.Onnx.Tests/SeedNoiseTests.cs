using Horus.F5Tts.Onnx;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Xunit;

namespace Horus.F5Tts.Onnx.Tests;

/// <summary>Pins the promise <see cref="F5TtsOptions.Seed"/> makes in its own docs: the same seed
/// reproduces the same audio, across machines and execution providers. That only holds if the noise
/// generator is deterministic and platform-independent — hence splitmix64 rather than
/// <see cref="Random"/>. These tests are the guard against someone "simplifying" it back.</summary>
public class SeedNoiseTests
{
    private static float[] Fill(int length, int seed)
    {
        var tensor = new DenseTensor<float>(new int[] { length });
        F5TtsModel.FillGaussian(tensor, seed);
        return tensor.ToArray();
    }

    private static Float16[] FillHalf(int length, int seed)
    {
        var tensor = new DenseTensor<Float16>(new int[] { length });
        F5TtsModel.FillGaussian(tensor, seed);
        return tensor.ToArray();
    }

    [Fact]
    public void FillGaussian_is_deterministic_for_the_same_seed()
    {
        Assert.Equal(Fill(256, 42), Fill(256, 42));
    }

    [Fact]
    public void FillGaussian_differs_between_seeds()
    {
        Assert.NotEqual(Fill(256, 42), Fill(256, 43));
    }

    [Fact]
    public void FillGaussian_produces_the_same_values_for_a_negative_seed_every_time()
    {
        // The seed is cast through uint into the generator state; negative values must stay stable
        // rather than throwing or collapsing to a constant.
        Assert.Equal(Fill(128, -7), Fill(128, -7));
        Assert.NotEqual(Fill(128, -7), Fill(128, 7));
    }

    [Fact]
    public void FillGaussian_fills_every_slot()
    {
        var values = Fill(512, 1);

        Assert.Equal(512, values.Length);
        Assert.DoesNotContain(values, v => float.IsNaN(v) || float.IsInfinity(v));
        // A constant fill would mean the generator never advanced.
        Assert.True(values.Distinct().Count() > 400);
    }

    [Fact]
    public void FillGaussian_is_deterministic_for_half_precision_too()
    {
        // FP16 exports need the same promise: a seed reproduces the audio.
        Assert.Equal(FillHalf(256, 42), FillHalf(256, 42));
        Assert.NotEqual(FillHalf(256, 42), FillHalf(256, 43));
    }

    [Fact]
    public void FillGaussian_draws_the_same_sequence_for_half_as_for_single()
    {
        // Both precisions must share one generator: the half values are simply the single values
        // rounded to the nearest representable half. If anyone ever gives the FP16 path an RNG of its
        // own, the two sequences drift and this catches it — a divergence that would otherwise only
        // show up as "the FP16 audio sounds subtly different than it should", i.e. never.
        var single = Fill(128, 42);
        var half = FillHalf(128, 42);

        for (var i = 0; i < single.Length; i++)
        {
            Assert.Equal((float)(Float16)single[i], (float)half[i]);
        }
    }

    [Fact]
    public void FillGaussian_is_approximately_standard_normal()
    {
        // Box-Muller over splitmix64 should give mean ~0 / sd ~1. Deterministic seed, so this is a
        // real check, not a flaky statistical one.
        var values = Fill(20_000, 12345);

        var mean = values.Average();
        var variance = values.Sum(v => (v - mean) * (v - mean)) / values.Length;
        var sd = Math.Sqrt(variance);

        Assert.InRange(mean, -0.05, 0.05);
        Assert.InRange(sd, 0.95, 1.05);
    }
}
