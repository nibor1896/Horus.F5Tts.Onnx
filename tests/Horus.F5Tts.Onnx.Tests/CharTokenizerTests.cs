using Horus.F5Tts.Onnx;
using Xunit;

namespace Horus.F5Tts.Onnx.Tests;

public class CharTokenizerTests
{
    private static readonly Dictionary<string, int> Vocab = new()
    {
        ["a"] = 1,
        ["b"] = 2,
        ["c"] = 3,
        ["ä"] = 4,
    };

    [Fact]
    public void Encode_maps_each_character_to_its_vocab_index()
    {
        var ids = new CharTokenizer(Vocab).Encode("abc");

        Assert.Equal([1, 2, 3], ids);
    }

    [Fact]
    public void Encode_maps_unknown_characters_to_the_filler_token()
    {
        // Index 0 is the filler. F5 vocabs typically do not list the space, so it lands here too —
        // that is by design, not a miss.
        var ids = new CharTokenizer(Vocab).Encode("a z");

        Assert.Equal([1, 0, 0], ids);
    }

    [Fact]
    public void Encode_handles_multi_byte_characters_present_in_the_vocab()
    {
        var ids = new CharTokenizer(Vocab).Encode("ä");

        Assert.Equal([4], ids);
    }

    [Fact]
    public void Encode_returns_empty_for_empty_text()
    {
        Assert.Empty(new CharTokenizer(Vocab).Encode(string.Empty));
    }

    [Fact]
    public void Encode_returns_one_id_per_character()
    {
        var ids = new CharTokenizer(Vocab).Encode("aabbcc");

        Assert.Equal(6, ids.Length);
    }

    [Fact]
    public void Constructor_rejects_a_null_vocabulary()
    {
        Assert.Throws<ArgumentNullException>(() => new CharTokenizer(null!));
    }
}
