using System.Text;
using Horus.F5Tts.Onnx;
using Xunit;

namespace Horus.F5Tts.Onnx.Tests;

public class TextChunkerTests
{
    [Fact]
    public void Split_returns_nothing_for_empty_or_whitespace_text()
    {
        Assert.Empty(TextChunker.Split(string.Empty));
        Assert.Empty(TextChunker.Split("   \r\n  "));
    }

    [Fact]
    public void Split_keeps_short_text_as_a_single_chunk()
    {
        var chunks = TextChunker.Split("Hello there. How are you?", 200);

        Assert.Single(chunks);
        Assert.Equal("Hello there. How are you?", chunks[0]);
    }

    [Fact]
    public void Split_packs_as_many_sentences_into_a_chunk_as_fit()
    {
        // Three ~10-byte sentences with a 25-byte budget: two fit, the third starts a new chunk.
        var chunks = TextChunker.Split("Aaa aaaa. Bbb bbbb. Ccc cccc.", 25);

        Assert.Equal(2, chunks.Count);
        Assert.Equal("Aaa aaaa. Bbb bbbb.", chunks[0]);
        Assert.Equal("Ccc cccc.", chunks[1]);
    }

    [Fact]
    public void Split_breaks_at_sentence_boundaries_not_mid_sentence()
    {
        var chunks = TextChunker.Split("First one here. Second one here. Third one here.", 20);

        Assert.All(chunks, c => Assert.Matches(@"[.!?]$", c));
    }

    [Fact]
    public void Split_keeps_a_single_over_long_sentence_whole()
    {
        // Cutting a sentence in half would be worse than exceeding the budget: the model would speak a
        // fragment. Better one long pass than two broken ones.
        var sentence = new string('a', 300) + ".";

        var chunks = TextChunker.Split(sentence, 50);

        Assert.Single(chunks);
        Assert.Equal(sentence, chunks[0]);
    }

    [Fact]
    public void Split_measures_the_budget_in_utf8_bytes_not_characters()
    {
        // "ä" is one char but two bytes — a byte budget must notice.
        var text = "ää ää. ää ää. ää ää.";

        var byBytes = TextChunker.Split(text, 14);

        Assert.True(byBytes.Count > 1, "a 14-byte budget must not swallow 20 bytes of text");
        Assert.All(byBytes, c => Assert.True(
            Encoding.UTF8.GetByteCount(c) <= 14 || !c.Contains(' '),
            $"chunk '{c}' is {Encoding.UTF8.GetByteCount(c)} bytes"));
    }

    [Fact]
    public void Split_breaks_after_full_width_cjk_punctuation()
    {
        // CJK sentences are not followed by a space, so the boundary is the punctuation itself.
        var chunks = TextChunker.Split("你好。世界。再见。", 9);

        Assert.True(chunks.Count > 1, "expected a break after the full-width stops");
    }

    [Fact]
    public void Split_splits_on_commas_and_semicolons_too()
    {
        // The reference implementation treats these as boundaries as well — with a tight budget they
        // are the only places long prose can be broken.
        var chunks = TextChunker.Split("one thing, two thing; three thing, four thing", 15);

        Assert.True(chunks.Count > 1);
    }

    [Fact]
    public void Split_rejects_a_non_positive_budget()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => TextChunker.Split("Hello.", 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => TextChunker.Split("Hello.", -1));
    }

    [Fact]
    public void MaxBytesFor_scales_with_the_references_speaking_rate()
    {
        // A reference that says more per second leaves room for more text per chunk.
        var slow = TextChunker.MaxBytesFor("short", 5.0);
        var fast = TextChunker.MaxBytesFor("a much longer transcript in the same five seconds", 5.0);

        Assert.True(fast > slow, $"expected {fast} > {slow}");
    }

    [Fact]
    public void MaxBytesFor_shrinks_as_the_reference_clip_eats_the_budget()
    {
        // The pass generates reference + new speech together, so a longer reference leaves less room.
        var shortRef = TextChunker.MaxBytesFor("some transcript", 3.0);
        var longRef = TextChunker.MaxBytesFor("some transcript", 15.0);

        Assert.True(longRef < shortRef, $"expected {longRef} < {shortRef}");
    }

    [Fact]
    public void MaxBytesFor_allows_more_text_at_higher_speed()
    {
        var normal = TextChunker.MaxBytesFor("some transcript", 5.0);
        var faster = TextChunker.MaxBytesFor("some transcript", 5.0, 1.5f);

        Assert.True(faster > normal, $"expected {faster} > {normal}");
    }

    [Theory]
    [InlineData(0)]      // nonsense duration
    [InlineData(-1)]
    [InlineData(22)]     // the reference alone already fills the pass
    [InlineData(60)]
    public void MaxBytesFor_falls_back_when_no_budget_can_be_derived(double referenceSeconds)
    {
        Assert.Equal(TextChunker.DefaultMaxBytes, TextChunker.MaxBytesFor("some transcript", referenceSeconds));
    }

    [Fact]
    public void MaxBytesFor_never_returns_a_useless_budget()
    {
        // An almost-empty transcript would otherwise round down to zero and make Split throw.
        Assert.True(TextChunker.MaxBytesFor("a", 21.9) >= 1);
    }
}
