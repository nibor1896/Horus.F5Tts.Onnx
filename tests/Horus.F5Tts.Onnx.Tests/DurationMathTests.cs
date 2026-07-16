using Horus.F5Tts.Onnx;
using Xunit;

namespace Horus.F5Tts.Onnx.Tests;

/// <summary>The duration/text maths that decides how long the transformer generates for. It is pure
/// arithmetic, but getting it wrong shows up as clipped or padded speech, so it is pinned here.</summary>
public class DurationMathTests
{
    // 25600 samples / hop 256 = 100 frames, +1 => 101 reference frames.
    private const int RefSamples = 25600;
    private const int RefFrames = 101;

    [Fact]
    public void ComputeMaxDuration_is_reference_frames_plus_scaled_text_plus_tail_pad()
    {
        // 101 + (long)(101/6 * 3) + 12  =>  101 + 50 + 12
        var d = F5TtsModel.ComputeMaxDuration(RefSamples, "abcdef", "abc", speed: 1.0f, tailPaddingFrames: 12);

        Assert.Equal(163, d);
    }

    [Fact]
    public void ComputeMaxDuration_grows_with_longer_generation_text()
    {
        var shortText = F5TtsModel.ComputeMaxDuration(RefSamples, "abcdef", "abc", 1.0f, 0);
        var longText = F5TtsModel.ComputeMaxDuration(RefSamples, "abcdef", "abcabcabc", 1.0f, 0);

        Assert.True(longText > shortText, $"expected {longText} > {shortText}");
    }

    [Fact]
    public void ComputeMaxDuration_shrinks_as_speed_rises()
    {
        var normal = F5TtsModel.ComputeMaxDuration(RefSamples, "abcdef", "abcabc", 1.0f, 0);
        var faster = F5TtsModel.ComputeMaxDuration(RefSamples, "abcdef", "abcabc", 2.0f, 0);

        Assert.True(faster < normal, $"expected {faster} < {normal}");
    }

    [Theory]
    [InlineData(0f)]
    [InlineData(-1f)]
    public void ComputeMaxDuration_treats_a_non_positive_speed_as_1(float speed)
    {
        var fallback = F5TtsModel.ComputeMaxDuration(RefSamples, "abcdef", "abc", speed, 0);
        var explicitOne = F5TtsModel.ComputeMaxDuration(RefSamples, "abcdef", "abc", 1.0f, 0);

        Assert.Equal(explicitOne, fallback);
    }

    [Fact]
    public void ComputeMaxDuration_adds_the_tail_pad_verbatim()
    {
        var without = F5TtsModel.ComputeMaxDuration(RefSamples, "abcdef", "abc", 1.0f, 0);
        var with = F5TtsModel.ComputeMaxDuration(RefSamples, "abcdef", "abc", 1.0f, 12);

        Assert.Equal(without + 12, with);
    }

    [Fact]
    public void ComputeMaxDuration_clamps_a_negative_tail_pad_to_zero()
    {
        var negative = F5TtsModel.ComputeMaxDuration(RefSamples, "abcdef", "abc", 1.0f, -50);
        var zero = F5TtsModel.ComputeMaxDuration(RefSamples, "abcdef", "abc", 1.0f, 0);

        Assert.Equal(zero, negative);
    }

    [Fact]
    public void ComputeMaxDuration_survives_empty_reference_text()
    {
        // refTextLen is clamped to 1, so this must not divide by zero.
        var d = F5TtsModel.ComputeMaxDuration(RefSamples, string.Empty, "abc", 1.0f, 0);

        Assert.True(d > RefFrames);
    }

    [Fact]
    public void ComputeMaxDuration_counts_text_in_utf8_bytes_not_chars()
    {
        // "ää" is 2 chars but 4 UTF-8 bytes — the model budgets per byte.
        var ascii = F5TtsModel.ComputeMaxDuration(RefSamples, "abcdef", "aa", 1.0f, 0);
        var umlauts = F5TtsModel.ComputeMaxDuration(RefSamples, "abcdef", "ää", 1.0f, 0);

        Assert.True(umlauts > ascii, $"expected {umlauts} > {ascii}");
    }

    [Fact]
    public void NormalizeReferenceText_appends_a_space_after_a_single_byte_ending()
    {
        Assert.Equal("Hello ", F5TtsModel.NormalizeReferenceText("Hello"));
    }

    [Fact]
    public void NormalizeReferenceText_leaves_an_existing_trailing_space_alone()
    {
        Assert.Equal("Hello ", F5TtsModel.NormalizeReferenceText("Hello "));
    }

    [Fact]
    public void NormalizeReferenceText_leaves_empty_text_alone()
    {
        Assert.Equal(string.Empty, F5TtsModel.NormalizeReferenceText(string.Empty));
    }

    [Theory]
    [InlineData("Grüße für dich ä")]  // 2-byte ending
    [InlineData("你好。")]              // 3-byte ending
    public void NormalizeReferenceText_leaves_multi_byte_endings_alone(string text)
    {
        Assert.Equal(text, F5TtsModel.NormalizeReferenceText(text));
    }
}
