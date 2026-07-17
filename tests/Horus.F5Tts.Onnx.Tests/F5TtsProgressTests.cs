using Horus.F5Tts.Onnx;
using Xunit;

namespace Horus.F5Tts.Onnx.Tests;

public class F5TtsProgressTests
{
    [Fact]
    public void Fraction_is_zero_before_the_first_step()
    {
        Assert.Equal(0, new F5TtsProgress(0, 1, 0, 31).Fraction);
    }

    [Fact]
    public void Fraction_is_one_when_the_only_chunk_finishes()
    {
        Assert.Equal(1.0, new F5TtsProgress(0, 1, 31, 31).Fraction, 9);
    }

    [Fact]
    public void Fraction_is_one_only_when_the_LAST_chunk_finishes()
    {
        // Finishing chunk 0 of 3 is a third of the way, not done. A bar that hit 100% at every
        // sentence would be worse than no bar at all.
        Assert.Equal(1.0 / 3, new F5TtsProgress(0, 3, 31, 31).Fraction, 9);
        Assert.Equal(2.0 / 3, new F5TtsProgress(1, 3, 31, 31).Fraction, 9);
        Assert.Equal(1.0, new F5TtsProgress(2, 3, 31, 31).Fraction, 9);
    }

    [Fact]
    public void Fraction_spans_the_whole_request_not_the_current_chunk()
    {
        // Half way through the second of four chunks = 37.5% overall, not 50%.
        Assert.Equal(0.375, new F5TtsProgress(1, 4, 16, 32).Fraction, 9);
    }

    [Fact]
    public void Fraction_advances_within_a_chunk()
    {
        var early = new F5TtsProgress(0, 1, 5, 31).Fraction;
        var later = new F5TtsProgress(0, 1, 20, 31).Fraction;

        Assert.True(later > early, $"expected {later} > {early}");
    }

    [Theory]
    [InlineData(0, 0, 0, 0)]    // nothing known
    [InlineData(0, 0, 5, 31)]   // no chunks
    [InlineData(0, 1, 5, 0)]    // no steps
    public void Fraction_is_zero_rather_than_NaN_when_the_counts_are_degenerate(
        int chunk, int chunkCount, int step, int stepCount)
    {
        // A divide-by-zero here would surface as a NaN width on someone's progress bar.
        Assert.Equal(0, new F5TtsProgress(chunk, chunkCount, step, stepCount).Fraction);
    }

    [Fact]
    public void Fraction_never_leaves_the_zero_to_one_range()
    {
        Assert.Equal(1.0, new F5TtsProgress(5, 3, 99, 31).Fraction, 9);
        Assert.Equal(0.0, new F5TtsProgress(-1, 3, -5, 31).Fraction, 9);
    }

    [Fact]
    public void Progress_carries_the_counters_for_a_label_not_just_a_number()
    {
        // "sentence 2 of 7" is what a UI actually writes next to the bar.
        var p = new F5TtsProgress(1, 7, 12, 31);

        Assert.Equal(1, p.Chunk);
        Assert.Equal(7, p.ChunkCount);
        Assert.Equal(12, p.Step);
        Assert.Equal(31, p.StepCount);
    }
}
