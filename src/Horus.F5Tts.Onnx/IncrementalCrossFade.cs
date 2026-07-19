namespace Horus.F5Tts.Onnx;

/// <summary>Cross-fades a sequence of audio segments <b>incrementally</b>, so each segment's final
/// audio can be emitted as soon as the next one exists — the engine behind
/// <see cref="F5TtsModel.SynthesizeStreamAsync"/>.
///
/// It is byte-for-byte equivalent to <see cref="F5TtsModel.CrossFade"/>: feed the same segments in the
/// same order and concatenate the returns, and you get the identical array. The trick is the
/// invariant below — <see cref="_pending"/> always holds exactly the last <c>fade</c> samples of the
/// result so far, and everything before it is final. A future blend only ever touches the last
/// <c>fade</c> samples (the overlap is capped at <c>fade</c>), so anything older can never change
/// again and is safe to emit. That is why streaming does not need a second cross-fade implementation
/// and cannot drift from the batch one — the two are bound by a bit-identity test.</summary>
internal sealed class IncrementalCrossFade(int fadeSamples)
{
    private short[] _pending = [];   // the last min(fade, total-so-far) samples; still blend-reachable
    private long _accumulated;       // total samples emitted-or-pending so far

    /// <summary>Feeds one more segment and returns the audio that is now final (may be empty). Pass
    /// <paramref name="isLast"/> = true for the final segment, which flushes the held tail.</summary>
    public short[] Next(short[] segment, bool isLast)
    {
        // The overlap blended between the tail so far and this segment's head — capped exactly as the
        // batch cross-fade caps it, so the blended samples match sample-for-sample.
        var overlap = (int)Math.Min(fadeSamples, Math.Min(_accumulated, segment.Length));

        // merged = [ pending, its last `overlap` samples blended with the segment head ] + segment[overlap..]
        var merged = new short[_pending.Length + segment.Length - overlap];
        Array.Copy(_pending, 0, merged, 0, _pending.Length - overlap);
        for (var i = 0; i < overlap; i++)
        {
            var t = (i + 1.0) / (overlap + 1.0);
            var blended = _pending[_pending.Length - overlap + i] * (1 - t) + segment[i] * t;
            merged[_pending.Length - overlap + i] =
                (short)Math.Clamp(Math.Round(blended), short.MinValue, short.MaxValue);
        }
        Array.Copy(segment, overlap, merged, _pending.Length, segment.Length - overlap);

        _accumulated += segment.Length - overlap;

        if (isLast)
        {
            _pending = [];
            return merged;
        }

        // Hold back the last `fade` samples (they may still be blended with the next segment); emit the rest.
        var keep = (int)Math.Min(fadeSamples, _accumulated);
        var emit = merged[..^keep];
        _pending = merged[^keep..];
        return emit;
    }
}
