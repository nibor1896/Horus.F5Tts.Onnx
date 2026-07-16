using Horus.F5Tts.Onnx;
using Xunit;

namespace Horus.F5Tts.Onnx.Tests;

public class VocabTests : IDisposable
{
    private readonly List<string> _temp = [];

    private string WriteVocab(string content)
    {
        var path = Path.Combine(Path.GetTempPath(), $"vocab-{Guid.NewGuid():N}.txt");
        File.WriteAllText(path, content);
        _temp.Add(path);
        return path;
    }

    public void Dispose()
    {
        foreach (var p in _temp)
        {
            File.Delete(p);
        }

        GC.SuppressFinalize(this);
    }

    [Fact]
    public void LoadVocab_maps_every_line_to_its_line_number()
    {
        var map = F5TtsModel.LoadVocab(WriteVocab("a\nb\nc"));

        Assert.Equal(0, map["a"]);
        Assert.Equal(1, map["b"]);
        Assert.Equal(2, map["c"]);
        Assert.Equal(3, map.Count);
    }

    [Fact]
    public void LoadVocab_handles_crlf_line_endings()
    {
        // The real vocab.txt may arrive with Windows line endings; the \r must not become part of
        // the token, or every lookup silently misses and maps to the filler.
        var map = F5TtsModel.LoadVocab(WriteVocab("a\r\nb\r\nc"));

        Assert.Equal(0, map["a"]);
        Assert.Equal(1, map["b"]);
        Assert.Equal(2, map["c"]);
    }

    [Fact]
    public void LoadVocab_ignores_the_empty_element_after_a_trailing_newline()
    {
        var map = F5TtsModel.LoadVocab(WriteVocab("a\nb\n"));

        Assert.Equal(2, map.Count);
        Assert.False(map.ContainsKey(""));
    }

    [Fact]
    public void LoadVocab_keeps_a_deliberate_empty_line_inside_the_file()
    {
        // Only the trailing element is dropped — an empty line in the middle is a real index.
        var map = F5TtsModel.LoadVocab(WriteVocab("a\n\nc\n"));

        Assert.Equal(0, map["a"]);
        Assert.Equal(1, map[""]);
        Assert.Equal(2, map["c"]);
    }

    [Fact]
    public void LoadVocab_reads_multi_byte_tokens()
    {
        var map = F5TtsModel.LoadVocab(WriteVocab("ä\nö\n漢\n"));

        Assert.Equal(0, map["ä"]);
        Assert.Equal(1, map["ö"]);
        Assert.Equal(2, map["漢"]);
    }
}
