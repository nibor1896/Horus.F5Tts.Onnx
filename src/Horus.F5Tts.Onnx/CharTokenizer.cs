namespace Horus.F5Tts.Onnx;

/// <summary>Character-level tokenizer: each character is looked up in the vocabulary; unknown
/// characters (including spaces, which F5 vocabs typically don't list) map to index 0, the filler
/// token. Verified end-to-end against the German F5-TTS ONNX export — no word segmentation
/// (jieba/pinyin) is required for Latin-script text.</summary>
public sealed class CharTokenizer : IF5Tokenizer
{
    private readonly IReadOnlyDictionary<string, int> _vocab;

    public CharTokenizer(IReadOnlyDictionary<string, int> vocab)
    {
        _vocab = vocab ?? throw new ArgumentNullException(nameof(vocab));
    }

    public int[] Encode(string text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return [];
        }

        var ids = new int[text.Length];
        for (var i = 0; i < text.Length; i++)
        {
            ids[i] = _vocab.TryGetValue(text[i].ToString(), out var id) ? id : 0;
        }

        return ids;
    }
}
