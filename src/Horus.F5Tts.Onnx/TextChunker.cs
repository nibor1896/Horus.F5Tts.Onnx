using System.Text;
using System.Text.RegularExpressions;

namespace Horus.F5Tts.Onnx;

/// <summary>Splits long text into pieces a single F5-TTS pass can handle.
///
/// Why this is needed at all: the model generates the reference audio <i>and</i> the new speech in one
/// go, and quality falls apart once that combined length runs much past ~22 seconds. So the limit is
/// not "how long is your text" but "how much time is left after the reference clip has had its share".
///
/// The rules here mirror the reference F5-TTS implementation rather than inventing their own, so a
/// given text chunks the same way in .NET as it does in Python.</summary>
public static partial class TextChunker
{
    /// <summary>The combined reference + generated budget, in seconds, that a single pass targets.
    /// From the reference implementation.</summary>
    private const double TotalBudgetSeconds = 22.0;

    /// <summary>Fallback chunk size in UTF-8 bytes when the budget cannot be derived from a reference
    /// clip. The reference implementation's default.</summary>
    public const int DefaultMaxBytes = 135;

    /// <summary>Sentence boundaries: after `;:,.!?` when whitespace follows, or straight after their
    /// full-width CJK counterparts (which are not followed by a space).</summary>
    [GeneratedRegex(@"(?<=[;:,.!?])\s+|(?<=[；：，。！？])")]
    private static partial Regex BoundaryRegex();

    /// <summary>How many UTF-8 bytes of new text still fit alongside this reference clip.
    ///
    /// Measures the reference's own speaking rate (bytes per second) and scales it by the time left in
    /// the budget once the reference has taken its share — so a long reference clip permits shorter
    /// chunks, which is exactly the trade the model imposes.</summary>
    /// <param name="referenceText">The transcript of the reference clip.</param>
    /// <param name="referenceSeconds">The reference clip's duration.</param>
    /// <param name="speed">The speaking rate the synthesis will use; faster speech fits more text.</param>
    /// <returns>A byte budget, never below 1.</returns>
    public static int MaxBytesFor(string referenceText, double referenceSeconds, float speed = 1.0f)
    {
        ArgumentNullException.ThrowIfNull(referenceText);

        if (referenceSeconds <= 0 || referenceSeconds >= TotalBudgetSeconds)
        {
            // No usable budget to derive: either we were handed nonsense, or the reference clip alone
            // already fills the pass. Fall back rather than return something absurd.
            return DefaultMaxBytes;
        }

        var bytesPerSecond = Encoding.UTF8.GetByteCount(referenceText) / referenceSeconds;
        var rate = speed <= 0 ? 1.0f : speed;
        var budget = (int)(bytesPerSecond * (TotalBudgetSeconds - referenceSeconds) * rate);
        return Math.Max(1, budget);
    }

    /// <summary>Splits <paramref name="text"/> into chunks of at most <paramref name="maxBytes"/> UTF-8
    /// bytes, breaking only at sentence boundaries and packing as many sentences into each chunk as
    /// fit. A single sentence longer than the budget is kept whole rather than cut mid-phrase: a
    /// clipped sentence is a worse outcome than an over-long one.</summary>
    /// <param name="text">The text to split. Empty or whitespace yields an empty list.</param>
    /// <param name="maxBytes">Budget per chunk; see <see cref="MaxBytesFor"/>.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxBytes"/> is not positive.</exception>
    public static IReadOnlyList<string> Split(string text, int maxBytes = DefaultMaxBytes)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (maxBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxBytes), maxBytes, "Budget must be positive.");
        }

        var chunks = new List<string>();
        if (string.IsNullOrWhiteSpace(text))
        {
            return chunks;
        }

        var current = new StringBuilder();
        foreach (var raw in BoundaryRegex().Split(text))
        {
            var sentence = raw.Trim();
            if (sentence.Length == 0)
            {
                continue;
            }

            if (current.Length == 0)
            {
                current.Append(sentence);
                continue;
            }

            var combined = Encoding.UTF8.GetByteCount(current.ToString()) + 1 + Encoding.UTF8.GetByteCount(sentence);
            if (combined <= maxBytes)
            {
                current.Append(' ').Append(sentence);
            }
            else
            {
                chunks.Add(current.ToString());
                current.Clear().Append(sentence);
            }
        }

        if (current.Length > 0)
        {
            chunks.Add(current.ToString());
        }

        return chunks;
    }
}
