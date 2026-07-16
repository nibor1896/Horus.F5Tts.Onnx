using Horus.F5Tts.Onnx;
using Xunit;

namespace Horus.F5Tts.Onnx.Tests;

public class CrossFadeTests
{
    private const int Rate = 24000;

    private static short[] Constant(int length, short value)
    {
        var s = new short[length];
        Array.Fill(s, value);
        return s;
    }

    [Fact]
    public void CrossFade_of_nothing_is_nothing()
    {
        Assert.Empty(F5TtsModel.CrossFade([], Rate, 0.15));
    }

    [Fact]
    public void CrossFade_of_a_single_segment_returns_it_unchanged()
    {
        var only = Constant(100, 500);

        Assert.Equal(only, F5TtsModel.CrossFade([only], Rate, 0.15));
    }

    [Fact]
    public void CrossFade_with_zero_overlap_just_concatenates()
    {
        var result = F5TtsModel.CrossFade([Constant(100, 1), Constant(50, 2)], Rate, 0);

        Assert.Equal(150, result.Length);
        Assert.Equal(1, result[99]);
        Assert.Equal(2, result[100]);
    }

    [Fact]
    public void CrossFade_shortens_the_result_by_the_overlap()
    {
        // 0.01 s at 24 kHz = 240 samples of overlap; the seam is shared, not duplicated.
        var result = F5TtsModel.CrossFade([Constant(1000, 100), Constant(1000, 100)], Rate, 0.01);

        Assert.Equal(1000 + 1000 - 240, result.Length);
    }

    [Fact]
    public void CrossFade_preserves_the_level_when_both_sides_are_equal()
    {
        // The whole point of a linear ramp: the two gains sum to 1, so blending a constant with the
        // same constant must leave it untouched. A ramp that dipped here would be audible as a hole
        // at every seam.
        var result = F5TtsModel.CrossFade([Constant(1000, 1000), Constant(1000, 1000)], Rate, 0.01);

        Assert.All(result, v => Assert.InRange(v, 999, 1001));
    }

    [Fact]
    public void CrossFade_ramps_from_one_segment_to_the_other()
    {
        var result = F5TtsModel.CrossFade([Constant(1000, 1000), Constant(1000, 0)], Rate, 0.01);

        const int seamStart = 1000 - 240;
        Assert.Equal(1000, result[seamStart - 1]);     // still fully the first segment
        Assert.True(result[seamStart] < 1000);          // ramping down
        Assert.True(result[seamStart + 239] < result[seamStart]); // monotonically, not abruptly
        Assert.Equal(0, result[1000]);                  // fully the second segment
    }

    [Fact]
    public void CrossFade_clamps_an_overlap_longer_than_the_segments()
    {
        // A 1 s fade requested over 10-sample clips must not read out of bounds.
        var result = F5TtsModel.CrossFade([Constant(10, 100), Constant(10, 100)], Rate, 1.0);

        Assert.Equal(10, result.Length); // fully overlapped
        Assert.All(result, v => Assert.InRange(v, 99, 101));
    }

    [Fact]
    public void CrossFade_joins_many_segments()
    {
        var segments = Enumerable.Range(0, 5).Select(_ => Constant(1000, 500)).ToArray();

        var result = F5TtsModel.CrossFade(segments, Rate, 0.01);

        Assert.Equal(5 * 1000 - 4 * 240, result.Length);
        Assert.All(result, v => Assert.InRange(v, 499, 501));
    }
}
