using Horus.F5Tts.Onnx;
using Xunit;

namespace Horus.F5Tts.Onnx.Tests;

/// <summary>The load-bearing guarantee of streaming: feeding segments to <see cref="IncrementalCrossFade"/>
/// one at a time and concatenating the pieces produces <b>exactly</b> the same array as the batch
/// <see cref="F5TtsModel.CrossFade"/>. This is what lets <c>SynthesizeStreamAsync</c> be "the batch
/// result, delivered incrementally" rather than a second rendering — and it needs no models, so it is
/// a fast, deterministic unit test.</summary>
public class StreamCrossFadeTests
{
    private const int Rate = 24000;
    private const double Fade = 0.15;                       // the library default
    private static int FadeSamples => (int)(Fade * Rate);   // exactly how CrossFade derives it

    private static short[] Incremental(IReadOnlyList<short[]> segments)
    {
        var fader = new IncrementalCrossFade(FadeSamples);
        var pieces = new List<short>();
        for (var i = 0; i < segments.Count; i++)
        {
            pieces.AddRange(fader.Next(segments[i], isLast: i == segments.Count - 1));
        }

        return pieces.ToArray();
    }

    private static short[] Random(int length, int seed)
    {
        var rnd = new Random(seed);
        var s = new short[length];
        for (var i = 0; i < length; i++)
        {
            s[i] = (short)rnd.Next(short.MinValue, short.MaxValue + 1);
        }

        return s;
    }

    [Fact]
    public void Streamed_pieces_concatenate_to_the_batch_result_bit_for_bit()
    {
        // Sentence-sized chunks (all comfortably longer than the fade) — the real regime.
        var segments = new[]
        {
            Random(40000, 1), Random(52000, 2), Random(33000, 3), Random(61000, 4), Random(28000, 5),
        };

        Assert.Equal(F5TtsModel.CrossFade(segments, Rate, Fade), Incremental(segments));
    }

    [Fact]
    public void A_single_segment_streams_unchanged()
    {
        var only = Random(30000, 9);

        Assert.Equal(F5TtsModel.CrossFade([only], Rate, Fade), Incremental([only]));
        Assert.Equal(only, Incremental([only]));
    }

    [Fact]
    public void Two_segments_match_the_batch()
    {
        var segments = new[] { Random(45000, 11), Random(45000, 12) };

        Assert.Equal(F5TtsModel.CrossFade(segments, Rate, Fade), Incremental(segments));
    }

    [Fact]
    public void Matches_the_batch_even_when_a_middle_segment_is_shorter_than_the_fade()
    {
        // Not something the sentence chunker produces, but the invariant must still hold: the held
        // tail is the last `fade` of the whole stream, not of one segment, so a short segment cannot
        // break the equivalence.
        var segments = new[] { Random(50000, 21), Random(FadeSamples / 2, 22), Random(50000, 23) };

        Assert.Equal(F5TtsModel.CrossFade(segments, Rate, Fade), Incremental(segments));
    }

    [Fact]
    public void Zero_fade_just_concatenates_like_the_batch()
    {
        var segments = new[] { Random(10000, 31), Random(10000, 32), Random(10000, 33) };
        var fader = new IncrementalCrossFade(0);
        var pieces = new List<short>();
        for (var i = 0; i < segments.Length; i++)
        {
            pieces.AddRange(fader.Next(segments[i], isLast: i == segments.Length - 1));
        }

        Assert.Equal(F5TtsModel.CrossFade(segments, Rate, 0), pieces.ToArray());
    }
}
