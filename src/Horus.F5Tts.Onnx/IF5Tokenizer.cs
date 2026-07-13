namespace Horus.F5Tts.Onnx;

/// <summary>Turns the combined reference+generation text into the integer token ids the model
/// expects. The default <see cref="CharTokenizer"/> maps each character to its vocab index, which
/// is sufficient for Latin-script languages (German, English, French, …). Chinese/Japanese
/// checkpoints need pinyin/jieba segmentation — implement this interface to plug that in.</summary>
public interface IF5Tokenizer
{
    /// <summary>Encodes already-combined text (reference text + a separating space + generation
    /// text) into token ids. The model applies its own +1 filler shift internally, so return the
    /// raw vocab indices.</summary>
    int[] Encode(string text);
}
