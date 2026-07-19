using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Horus.F5Tts.Onnx;

/// <summary>
/// Runs F5-TTS speech synthesis entirely through ONNX Runtime — no Python, no PyTorch. Load the
/// three exported models once (they are heavy, especially the transformer), then call
/// <see cref="Synthesize"/> as often as you like. Thread-safe: calls are serialized internally.
///
/// Expects the three-model export produced by <c>DakeQQ/F5-TTS-ONNX</c>:
///   Preprocess : reference audio (int16) + text ids + max_duration
///                → noise, RoPE tables (q/k), text/mel conditioning, ref_signal_len
///   Transformer: one denoising step of the DiT (driven in an NFE loop; time step in "time_step.1")
///   Decode     : denoised mel + ref_signal_len → 24 kHz int16 waveform
///
/// All heavy signal processing (STFT, mel, RoPE, classifier-free guidance, the ODE solver, the
/// vocoder) lives inside the ONNX graphs — this class only marshals tensors and runs the loop.
/// </summary>
public sealed class F5TtsModel : IDisposable
{
    private const int SampleRate = 24000;
    private const int HopLength = 256;

    private readonly InferenceSession _preprocess;
    private readonly InferenceSession _transformer;
    private readonly InferenceSession _decode;
    private readonly IReadOnlyDictionary<string, int> _vocab;
    private readonly CharTokenizer _defaultTokenizer;

    /// <summary>Whether this export works in half precision. Read from the model itself rather than
    /// asked of the caller: an FP16 export is self-consistent (its preprocess already emits
    /// <c>Float16</c>, which the transformer and decode then expect), so the tensors the pipeline
    /// marshals have to match, and nothing about that is the consumer's business.</summary>
    private readonly bool _isFloat16;
    /// <summary>Serializes the model work. Deliberately a semaphore rather than a <c>lock</c>: the
    /// async path can then queue without blocking a thread-pool thread, and a caller can still be
    /// cancelled while it waits its turn.
    ///
    /// Why serialize at all: ONNX Runtime does not document whether <c>Run</c> may be called
    /// concurrently on one session, and this workload is compute-bound anyway — parallel calls would
    /// share the same device and add no throughput, only contention.</summary>
    private readonly SemaphoreSlim _gate = new(1, 1);

    private F5TtsModel(InferenceSession preprocess, InferenceSession transformer, InferenceSession decode,
        IReadOnlyDictionary<string, int> vocab)
    {
        _preprocess = preprocess;
        _transformer = transformer;
        _decode = decode;
        _vocab = vocab;
        _defaultTokenizer = new CharTokenizer(vocab);
        _isFloat16 = preprocess.OutputMetadata.TryGetValue("noise", out var noiseMeta)
                     && noiseMeta.ElementType == typeof(Float16);
    }

    /// <summary>The audio sample rate of the synthesized output (24 kHz).</summary>
    public int OutputSampleRate => SampleRate;

    /// <summary>Binds a reference clip and its transcript into a <see cref="PreparedVoice"/>, so you
    /// can then synthesize with just the text (see <see cref="PreparedVoice"/> for what this does and
    /// does not save).</summary>
    /// <param name="referenceAudio">The reference voice, as 24 kHz mono 16-bit PCM.</param>
    /// <param name="referenceText">The transcript of <paramref name="referenceAudio"/>.</param>
    public PreparedVoice PrepareVoice(short[] referenceAudio, string referenceText)
    {
        ArgumentNullException.ThrowIfNull(referenceAudio);
        ArgumentNullException.ThrowIfNull(referenceText);
        return new PreparedVoice(this, referenceAudio, referenceText);
    }

    /// <summary>Loads a reference clip from a .wav (any rate, mono or stereo — resampled and
    /// down-mixed to 24 kHz for you) and binds it with its transcript into a <see cref="PreparedVoice"/>.</summary>
    /// <param name="wavPath">Path to the reference .wav.</param>
    /// <param name="referenceText">The transcript of the clip.</param>
    public PreparedVoice PrepareVoiceFromWav(string wavPath, string referenceText)
    {
        ArgumentNullException.ThrowIfNull(wavPath);
        ArgumentNullException.ThrowIfNull(referenceText);
        return new PreparedVoice(this, WavAudio.ReadPcm16Resampled(wavPath, SampleRate), referenceText);
    }

    /// <summary>Loads the three ONNX models and the vocabulary. Optionally configure the ONNX Runtime
    /// session (e.g. append a GPU execution provider). When <paramref name="configureSession"/> is
    /// null, the default CPU provider is used.</summary>
    /// <param name="preprocessModelPath">Path to <c>F5_Preprocess.onnx</c>.</param>
    /// <param name="transformerModelPath">Path to <c>F5_Transformer.onnx</c>.</param>
    /// <param name="decodeModelPath">Path to <c>F5_Decode.onnx</c>.</param>
    /// <param name="vocabPath">Path to <c>vocab.txt</c> (one token per line, in index order).</param>
    /// <param name="configureSession">Optional hook to set execution providers / options, e.g.
    /// <c>o =&gt; o.AppendExecutionProvider_DML(0)</c> (needs the Microsoft.ML.OnnxRuntime.DirectML
    /// package) or <c>o =&gt; o.AppendExecutionProvider_CUDA(0)</c> (needs the .Gpu package).</param>
    public static F5TtsModel Load(
        string preprocessModelPath,
        string transformerModelPath,
        string decodeModelPath,
        string vocabPath,
        Action<SessionOptions>? configureSession = null)
    {
        var vocab = LoadVocab(vocabPath);

        SessionOptions? options = null;
        InferenceSession? preprocess = null, transformer = null, decode = null;
        try
        {
            if (configureSession is not null)
            {
                options = new SessionOptions();
                configureSession(options);
            }

            preprocess = options is null ? new InferenceSession(preprocessModelPath) : new InferenceSession(preprocessModelPath, options);
            transformer = options is null ? new InferenceSession(transformerModelPath) : new InferenceSession(transformerModelPath, options);
            decode = options is null ? new InferenceSession(decodeModelPath) : new InferenceSession(decodeModelPath, options);
            return new F5TtsModel(preprocess, transformer, decode, vocab);
        }
        catch
        {
            preprocess?.Dispose();
            transformer?.Dispose();
            decode?.Dispose();
            throw;
        }
        finally
        {
            options?.Dispose();
        }
    }

    /// <summary>Synthesizes <paramref name="text"/> in the voice of the reference clip.</summary>
    /// <param name="referenceAudio">The reference voice, as 24 kHz mono 16-bit PCM. Use
    /// <see cref="WavAudio.ReadPcm16(string)"/> to load a .wav (already at 24 kHz mono).</param>
    /// <param name="referenceText">The transcript of <paramref name="referenceAudio"/>.</param>
    /// <param name="text">The text to speak.</param>
    /// <param name="options">Optional synthesis options (NFE steps, speed, custom tokenizer,
    /// text normalizer).</param>
    public F5TtsResult Synthesize(short[] referenceAudio, string referenceText, string text, F5TtsOptions? options = null)
    {
        var plan = Prepare(referenceAudio, referenceText, text, options);

        _gate.Wait();
        try
        {
            return Execute(plan, CancellationToken.None);
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <summary>Synthesizes on a background thread, so a UI or request thread is not blocked by the
    /// CPU/GPU-bound work. Identical to <see cref="Synthesize"/> in every other respect — same audio
    /// for the same inputs.</summary>
    /// <remarks>Cancellation is checked while queued, before the pipeline starts, and <b>between
    /// denoising steps</b>: the library drives that loop itself, so a request can be abandoned
    /// part-way instead of only before it begins. One step is the granularity — an ONNX Runtime call
    /// already in flight cannot be interrupted, so cancelling takes effect within roughly one step,
    /// which is on the order of a second for a large model.
    ///
    /// Calls against one instance are serialized. Waiting for your turn costs no thread-pool thread
    /// and does not delay cancellation.</remarks>
    /// <param name="referenceAudio">The reference voice, as 24 kHz mono 16-bit PCM.</param>
    /// <param name="referenceText">The transcript of <paramref name="referenceAudio"/>.</param>
    /// <param name="text">The text to speak.</param>
    /// <param name="options">Optional synthesis options.</param>
    /// <param name="cancellationToken">Cancels the synthesis; see the remarks for the granularity.</param>
    /// <exception cref="OperationCanceledException"><paramref name="cancellationToken"/> was cancelled.</exception>
    public async Task<F5TtsResult> SynthesizeAsync(short[] referenceAudio, string referenceText, string text,
        F5TtsOptions? options = null, CancellationToken cancellationToken = default)
    {
        var plan = Prepare(referenceAudio, referenceText, text, options);

        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            return await Task.Run(() => Execute(plan, cancellationToken), cancellationToken)
                .ConfigureAwait(false);
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <summary>Default overlap blended between consecutive chunks in <see cref="SynthesizeLong"/>.
    /// Matches the reference F5-TTS implementation.</summary>
    public const double DefaultCrossFadeSeconds = 0.15;

    /// <summary>Synthesizes text of any length by splitting it at sentence boundaries, synthesizing
    /// each piece and cross-fading the results together.
    ///
    /// Use this instead of <see cref="Synthesize"/> when the text may be long. A single pass generates
    /// the reference clip <i>and</i> the new speech together, and quality degrades once that combined
    /// length runs much past ~22 seconds — so the usable text budget depends on how much of the pass
    /// the reference clip already consumes. <see cref="TextChunker"/> works that budget out; short text
    /// stays a single pass and costs nothing extra.</summary>
    /// <param name="referenceAudio">The reference voice, as 24 kHz mono 16-bit PCM.</param>
    /// <param name="referenceText">The transcript of <paramref name="referenceAudio"/>.</param>
    /// <param name="text">The text to speak; may span many sentences.</param>
    /// <param name="options">Optional synthesis options, applied to every chunk. A <see
    /// cref="F5TtsOptions.Seed"/> keeps the whole result reproducible: each chunk derives its own seed
    /// from it, so the pieces get different noise — as they would within one pass — while the output as
    /// a whole repeats exactly.</param>
    /// <param name="crossFadeSeconds">Overlap blended between consecutive chunks. Zero butt-joins them,
    /// which tends to click.</param>
    public F5TtsResult SynthesizeLong(short[] referenceAudio, string referenceText, string text,
        F5TtsOptions? options = null, double crossFadeSeconds = DefaultCrossFadeSeconds)
    {
        ArgumentNullException.ThrowIfNull(referenceAudio);
        ArgumentNullException.ThrowIfNull(referenceText);
        ArgumentNullException.ThrowIfNull(text);
        options ??= new F5TtsOptions();

        var chunks = ChunkFor(referenceAudio, referenceText, text, options);
        if (chunks.Count <= 1)
        {
            return Synthesize(referenceAudio, referenceText, text, options);
        }

        var segments = new List<short[]>(chunks.Count);
        for (var i = 0; i < chunks.Count; i++)
        {
            segments.Add(Synthesize(
                referenceAudio, referenceText, chunks[i], OptionsForChunk(options, i, chunks.Count)).Samples);
        }

        return new F5TtsResult(CrossFade(segments, SampleRate, crossFadeSeconds), SampleRate);
    }

    /// <summary>Background-thread counterpart of <see cref="SynthesizeLong"/>. Cancellation is honoured
    /// between chunks as well as inside each one, so a long job stops promptly.</summary>
    public async Task<F5TtsResult> SynthesizeLongAsync(short[] referenceAudio, string referenceText, string text,
        F5TtsOptions? options = null, double crossFadeSeconds = DefaultCrossFadeSeconds,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(referenceAudio);
        ArgumentNullException.ThrowIfNull(referenceText);
        ArgumentNullException.ThrowIfNull(text);
        options ??= new F5TtsOptions();

        var chunks = ChunkFor(referenceAudio, referenceText, text, options);
        if (chunks.Count <= 1)
        {
            return await SynthesizeAsync(referenceAudio, referenceText, text, options, cancellationToken)
                .ConfigureAwait(false);
        }

        var segments = new List<short[]>(chunks.Count);
        for (var i = 0; i < chunks.Count; i++)
        {
            var result = await SynthesizeAsync(
                referenceAudio, referenceText, chunks[i], OptionsForChunk(options, i, chunks.Count),
                cancellationToken)
                .ConfigureAwait(false);
            segments.Add(result.Samples);
        }

        return new F5TtsResult(CrossFade(segments, SampleRate, crossFadeSeconds), SampleRate);
    }

    /// <summary>Synthesizes long text and <b>streams</b> the audio chunk by chunk, so the first sound
    /// is ready after the first sentence instead of after the whole text. For anything interactive —
    /// an assistant speaking a paragraph, a chat UI — the wait that matters is time-to-first-audio,
    /// and that drops from "synthesize everything" to "synthesize the first chunk".
    ///
    /// The chunks are the same sentence-level pieces <see cref="SynthesizeLong"/> makes, cross-faded
    /// the same way, so <b>concatenating every yielded <see cref="F5TtsChunk.Samples"/> in order gives
    /// exactly the same audio as <see cref="SynthesizeLongAsync"/></b> for the same inputs and seed. A
    /// typical consumer writes one WAV header (or opens one PCM audio sink) and appends each chunk as
    /// it arrives.</summary>
    /// <remarks>This is <b>chunk</b>-granularity streaming, not frame-level: F5-TTS generates each
    /// chunk's audio as a whole (a flow-matching model, not autoregressive), so sound cannot stream
    /// from <i>within</i> a chunk — the first chunk simply becomes available while the rest are still
    /// being generated. Short text that is a single chunk yields exactly one item, identical to
    /// <see cref="SynthesizeAsync"/>. Cancellation is honoured between and within chunks, as in
    /// <see cref="SynthesizeAsync"/>; any tail held back for the next cross-fade is simply dropped.</remarks>
    /// <param name="referenceAudio">The reference voice, as 24 kHz mono 16-bit PCM.</param>
    /// <param name="referenceText">The transcript of <paramref name="referenceAudio"/>.</param>
    /// <param name="text">The text to speak; may span many sentences.</param>
    /// <param name="options">Optional synthesis options, applied to every chunk (see
    /// <see cref="SynthesizeLong"/> for how a <see cref="F5TtsOptions.Seed"/> stays reproducible).</param>
    /// <param name="crossFadeSeconds">Overlap blended between consecutive chunks. Zero butt-joins them,
    /// which tends to click.</param>
    /// <param name="cancellationToken">Cancels the synthesis; stops the enumeration promptly.</param>
    public async IAsyncEnumerable<F5TtsChunk> SynthesizeStreamAsync(
        short[] referenceAudio, string referenceText, string text,
        F5TtsOptions? options = null, double crossFadeSeconds = DefaultCrossFadeSeconds,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(referenceAudio);
        ArgumentNullException.ThrowIfNull(referenceText);
        ArgumentNullException.ThrowIfNull(text);
        options ??= new F5TtsOptions();

        var chunks = ChunkFor(referenceAudio, referenceText, text, options);
        var fadeSamples = crossFadeSeconds <= 0 ? 0 : (int)(crossFadeSeconds * SampleRate);
        var fader = new IncrementalCrossFade(fadeSamples);

        for (var i = 0; i < chunks.Count; i++)
        {
            var result = await SynthesizeAsync(
                referenceAudio, referenceText, chunks[i], OptionsForChunk(options, i, chunks.Count),
                cancellationToken)
                .ConfigureAwait(false);

            var samples = fader.Next(result.Samples, isLast: i == chunks.Count - 1);
            yield return new F5TtsChunk(samples, SampleRate, i, chunks.Count, chunks[i]);
        }
    }

    private IReadOnlyList<string> ChunkFor(short[] referenceAudio, string referenceText, string text,
        F5TtsOptions options)
    {
        var referenceSeconds = referenceAudio.Length / (double)SampleRate;
        return TextChunker.Split(text, TextChunker.MaxBytesFor(referenceText, referenceSeconds, options.Speed));
    }

    /// <summary>Adapts one chunk's options: its own seed, and progress rewritten to say which chunk it
    /// is.
    ///
    /// The seed is derived rather than reused — reusing one verbatim would start every sentence from
    /// identical noise, which a single long pass would never do. Unseeded input is left alone, since it
    /// already varies per call.</summary>
    private static F5TtsOptions OptionsForChunk(F5TtsOptions options, int index, int count)
    {
        var needsSeed = options.Seed is not null && index > 0;
        var needsProgress = options.Progress is not null && count > 1;
        if (!needsSeed && !needsProgress)
        {
            return options;
        }

        return new F5TtsOptions
        {
            NfeSteps = options.NfeSteps,
            Speed = options.Speed,
            Tokenizer = options.Tokenizer,
            TextNormalizer = options.TextNormalizer,
            Seed = needsSeed ? unchecked(options.Seed!.Value + index) : options.Seed,
            TailPaddingFrames = options.TailPaddingFrames,
            Progress = needsProgress ? new ChunkProgress(options.Progress!, index, count) : options.Progress,
        };
    }

    /// <summary>Rewrites a single pass's "chunk 0 of 1" into its real position in a longer job, so the
    /// pipeline never has to know it is one piece of several — and the caller sees one continuous
    /// 0..1 instead of a bar that restarts at every sentence.</summary>
    private sealed class ChunkProgress(IProgress<F5TtsProgress> inner, int chunk, int chunkCount)
        : IProgress<F5TtsProgress>
    {
        public void Report(F5TtsProgress value)
            => inner.Report(new F5TtsProgress(chunk, chunkCount, value.Step, value.StepCount));
    }

    /// <summary>Joins segments, blending <paramref name="crossFadeSeconds"/> of overlap between each
    /// pair with a linear ramp. Butt-joining two independently generated waveforms leaves a
    /// discontinuity that is audible as a click, which is why the overlap exists at all.</summary>
    internal static short[] CrossFade(IReadOnlyList<short[]> segments, int sampleRate, double crossFadeSeconds)
    {
        if (segments.Count == 0)
        {
            return [];
        }

        var fadeSamples = crossFadeSeconds <= 0 ? 0 : (int)(crossFadeSeconds * sampleRate);
        var result = segments[0];

        for (var s = 1; s < segments.Count; s++)
        {
            var next = segments[s];
            var overlap = Math.Min(fadeSamples, Math.Min(result.Length, next.Length));

            if (overlap <= 0)
            {
                var joined = new short[result.Length + next.Length];
                result.CopyTo(joined, 0);
                next.CopyTo(joined, result.Length);
                result = joined;
                continue;
            }

            var combined = new short[result.Length + next.Length - overlap];
            Array.Copy(result, 0, combined, 0, result.Length - overlap);

            for (var i = 0; i < overlap; i++)
            {
                var t = (i + 1.0) / (overlap + 1.0);
                var blended = result[result.Length - overlap + i] * (1 - t) + next[i] * t;
                combined[result.Length - overlap + i] =
                    (short)Math.Clamp(Math.Round(blended), short.MinValue, short.MaxValue);
            }

            Array.Copy(next, overlap, combined, result.Length, next.Length - overlap);
            result = combined;
        }

        return result;
    }

    /// <summary>The prepared inputs for one synthesis, so the gate only has to cover the model work.</summary>
    private readonly record struct Plan(
        short[] ReferenceAudio, int[] TextIds, long MaxDuration, int NfeSteps, int? Seed,
        IProgress<F5TtsProgress>? Progress);

    /// <summary>Everything that can be done before taking the gate. Validation, normalization and
    /// tokenization are cheap and touch no shared state, so they must not hold up another caller —
    /// and a bad argument should be rejected immediately rather than after queuing.</summary>
    private Plan Prepare(short[] referenceAudio, string referenceText, string text, F5TtsOptions? options)
    {
        ArgumentNullException.ThrowIfNull(referenceAudio);
        ArgumentNullException.ThrowIfNull(referenceText);
        ArgumentNullException.ThrowIfNull(text);
        options ??= new F5TtsOptions();

        var genText = options.TextNormalizer is null ? text : options.TextNormalizer(text);
        var tokenizer = options.Tokenizer ?? _defaultTokenizer;

        var refPart = NormalizeReferenceText(referenceText);
        var textIds = tokenizer.Encode(refPart + genText);
        var maxDuration = ComputeMaxDuration(
            referenceAudio.Length, refPart, genText, options.Speed, options.TailPaddingFrames);

        return new Plan(referenceAudio, textIds, maxDuration, options.NfeSteps, options.Seed, options.Progress);
    }

    /// <summary>The model work itself. Only ever runs while the gate is held.</summary>
    private F5TtsResult Execute(Plan plan, CancellationToken cancellationToken)
    {
        var samples = _isFloat16
            ? RunPipeline<Float16>(plan.ReferenceAudio, plan.TextIds, plan.MaxDuration, plan.NfeSteps,
                plan.Seed, plan.Progress, cancellationToken)
            : RunPipeline<float>(plan.ReferenceAudio, plan.TextIds, plan.MaxDuration, plan.NfeSteps,
                plan.Seed, plan.Progress, cancellationToken);
        return new F5TtsResult(samples, SampleRate);
    }

    /// <summary>Appends a trailing space when the reference text ends on a single-byte character —
    /// without it the last reference word fuses with the first generated one. Multi-byte endings
    /// (CJK punctuation, umlauts, …) are left alone, as is text that already ends on a space.</summary>
    internal static string NormalizeReferenceText(string referenceText)
    {
        if (referenceText.Length > 0
            && Encoding.UTF8.GetByteCount(referenceText[^1].ToString()) == 1
            && !referenceText.EndsWith(' '))
        {
            return referenceText + " ";
        }

        return referenceText;
    }

    /// <summary>Target mel length the transformer generates into: the reference audio's own frame
    /// count, plus that same duration-per-text-byte scaled onto the generation text and divided by
    /// the speaking rate, plus the tail pad. Non-positive speeds fall back to 1.0 and a negative tail
    /// pad is clamped to 0.</summary>
    internal static long ComputeMaxDuration(int referenceSampleCount, string referenceTextPart,
        string generationText, float speed, int tailPaddingFrames)
    {
        var refTextLen = Math.Max(1, Encoding.UTF8.GetByteCount(referenceTextPart));
        var genTextLen = Encoding.UTF8.GetByteCount(generationText);
        var refAudioLen = referenceSampleCount / HopLength + 1;
        var rate = speed <= 0 ? 1.0 : speed;
        return refAudioLen
            + (long)((double)refAudioLen / refTextLen * genTextLen / rate)
            + Math.Max(0, tailPaddingFrames);
    }

    /// <summary>The pipeline, generic over the export's float element type: <c>float</c> for an F32
    /// export, <see cref="Float16"/> for a half-precision one. Nothing in here does arithmetic on the
    /// element — the tensors are only marshalled between the three models — so one body serves both
    /// rather than two copies that could drift apart.</summary>
    /// <typeparam name="T">Must match what this export uses; see <see cref="_isFloat16"/>.</typeparam>
    private short[] RunPipeline<T>(short[] referenceAudio, int[] textIds, long maxDuration, int nfeSteps, int? seed,
        IProgress<F5TtsProgress>? progress, CancellationToken cancellationToken)
        where T : struct
    {
        cancellationToken.ThrowIfCancellationRequested();

        var audioTensor = new DenseTensor<short>(referenceAudio, [1, 1, referenceAudio.Length]);
        var textTensor = new DenseTensor<int>(textIds, [1, textIds.Length]);
        var maxDurTensor = new DenseTensor<long>(new long[] { maxDuration }, new int[] { 1 });

        DenseTensor<T> noise, ropeCosQ, ropeSinQ, ropeCosK, ropeSinK, catMelText, catMelTextDrop;
        DenseTensor<long> refSignalLen;

        using (var pre = _preprocess.Run([
            NamedOnnxValue.CreateFromTensor("audio", audioTensor),
            NamedOnnxValue.CreateFromTensor("text_ids", textTensor),
            NamedOnnxValue.CreateFromTensor("max_duration", maxDurTensor),
        ]))
        {
            noise = Copy<T>(pre, "noise");
            ropeCosQ = Copy<T>(pre, "rope_cos_q");
            ropeSinQ = Copy<T>(pre, "rope_sin_q");
            ropeCosK = Copy<T>(pre, "rope_cos_k");
            ropeSinK = Copy<T>(pre, "rope_sin_k");
            catMelText = Copy<T>(pre, "cat_mel_text");
            catMelTextDrop = Copy<T>(pre, "cat_mel_text_drop");
            refSignalLen = Copy<long>(pre, "ref_signal_len");
        }

        // When a seed is requested, replace the model's freshly-drawn random noise with deterministic
        // seeded standard-normal noise of the same shape, so synthesis is reproducible (same
        // reference + text + seed → identical audio). Left untouched otherwise (natural F5 variation).
        // The generator is identical for both precisions; only the final conversion differs, so a seed
        // stays reproducible within a precision — across them the audio differs, as it must.
        if (seed is int s)
        {
            switch ((object)noise)
            {
                case DenseTensor<float> f:
                    FillGaussian(f, s);
                    break;
                case DenseTensor<Float16> h:
                    FillGaussian(h, s);
                    break;
                default:
                    throw new NotSupportedException($"Seeding is not supported for element type {typeof(T)}.");
            }
        }

        // The transformer performs NFE-1 denoising steps. time_step starts at 0 and the model
        // returns the next time_step to feed back on the following iteration — this matches
        // DakeQQ's F5-TTS-ONNX inference driver (range(0, NFE-1), feeding the returned time_step).
        var x = noise;
        var timeStep = new DenseTensor<int>(new int[] { 0 }, new int[] { 1 });
        for (var i = 0; i < nfeSteps - 1; i++)
        {
            // The only place a long synthesis can be abandoned: one step is in flight at a time and
            // ONNX Runtime cannot be interrupted mid-call, so this is the achievable granularity.
            cancellationToken.ThrowIfCancellationRequested();

            using var res = _transformer.Run([
                NamedOnnxValue.CreateFromTensor("noise", x),
                NamedOnnxValue.CreateFromTensor("rope_cos_q", ropeCosQ),
                NamedOnnxValue.CreateFromTensor("rope_sin_q", ropeSinQ),
                NamedOnnxValue.CreateFromTensor("rope_cos_k", ropeCosK),
                NamedOnnxValue.CreateFromTensor("rope_sin_k", ropeSinK),
                NamedOnnxValue.CreateFromTensor("cat_mel_text", catMelText),
                NamedOnnxValue.CreateFromTensor("cat_mel_text_drop", catMelTextDrop),
                NamedOnnxValue.CreateFromTensor("time_step.1", timeStep),
            ]);
            x = Copy<T>(res, "denoised");
            timeStep = Copy<int>(res, "time_step");

            // Always reported as chunk 0 of 1. SynthesizeLong wraps this in an adapter that rewrites
            // it to the real chunk, so the loop stays unaware of whether it is part of a longer job.
            progress?.Report(new F5TtsProgress(0, 1, i + 1, nfeSteps - 1));
        }

        cancellationToken.ThrowIfCancellationRequested();

        using var dec = _decode.Run([
            NamedOnnxValue.CreateFromTensor("denoised", x),
            NamedOnnxValue.CreateFromTensor("ref_signal_len", refSignalLen),
        ]);
        return dec.First(r => r.Name == "output_audio").AsTensor<short>().ToArray();
    }

    private static DenseTensor<T> Copy<T>(IReadOnlyCollection<DisposableNamedOnnxValue> results, string name)
    {
        var tensor = results.First(r => r.Name == name).AsTensor<T>();
        return new DenseTensor<T>(tensor.ToArray(), tensor.Dimensions.ToArray());
    }

    /// <summary>Fills a tensor in place with deterministic standard-normal noise (Box–Muller) driven
    /// by a splitmix64 generator. Deliberately not <see cref="Random"/>: a small, well-defined RNG
    /// reproduces bit-for-bit across platforms and languages, and avoids the occasional unlucky
    /// <see cref="Random"/> draw whose sequence tail can destabilize the denoiser on some execution
    /// providers.</summary>
    internal static void FillGaussian(DenseTensor<float> tensor, int seed)
    {
        var source = new GaussianSource(seed);
        var span = tensor.Buffer.Span;
        for (var i = 0; i < span.Length; i++)
        {
            span[i] = (float)source.Next();
        }
    }

    /// <summary>Half-precision counterpart for FP16 exports. Same generator, same draw order, so a
    /// seed is exactly as reproducible — the values simply land on the nearest representable half.
    /// That is also why the same seed yields different audio on an FP16 export than on an F32 one:
    /// less precision, different numbers, by design.</summary>
    internal static void FillGaussian(DenseTensor<Float16> tensor, int seed)
    {
        var source = new GaussianSource(seed);
        var span = tensor.Buffer.Span;
        for (var i = 0; i < span.Length; i++)
        {
            span[i] = (Float16)(float)source.Next();
        }
    }

    /// <summary>splitmix64 + Box–Muller, shared by both precisions so their draw sequences cannot
    /// drift apart.</summary>
    private struct GaussianSource
    {
        private const ulong Gamma = 0x9E3779B97F4A7C15UL;

        private ulong _state;

        public GaussianSource(int seed) => _state = (ulong)(uint)seed;

        /// <summary>One standard-normal draw. Consumes two uniforms, in that order.</summary>
        public double Next()
        {
            // 1.0 - NextDouble() is in (0,1], which avoids Log(0).
            var u1 = 1.0 - NextDouble();
            var u2 = 1.0 - NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }

        private double NextDouble()
        {
            _state += Gamma;
            var z = _state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            z ^= z >> 31;
            return (z >> 11) * (1.0 / (1UL << 53));
        }
    }

    /// <summary>Reads <c>vocab.txt</c> into a token → index map (one token per line, index = line
    /// number). Handles both LF and CRLF line endings.</summary>
    public static Dictionary<string, int> LoadVocab(string vocabPath)
    {
        var map = new Dictionary<string, int>();
        var lines = File.ReadAllText(vocabPath).Split('\n');
        for (var i = 0; i < lines.Length; i++)
        {
            var line = lines[i];
            if (line.EndsWith('\r'))
            {
                line = line[..^1];
            }

            // The empty element after a trailing newline is not a real token.
            if (i == lines.Length - 1 && line.Length == 0)
            {
                break;
            }

            map[line] = i;
        }

        return map;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _preprocess.Dispose();
        _transformer.Dispose();
        _decode.Dispose();
        _gate.Dispose();
    }
}
