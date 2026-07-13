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
    private readonly object _sync = new();

    private F5TtsModel(InferenceSession preprocess, InferenceSession transformer, InferenceSession decode,
        IReadOnlyDictionary<string, int> vocab)
    {
        _preprocess = preprocess;
        _transformer = transformer;
        _decode = decode;
        _vocab = vocab;
        _defaultTokenizer = new CharTokenizer(vocab);
    }

    /// <summary>The audio sample rate of the synthesized output (24 kHz).</summary>
    public int OutputSampleRate => SampleRate;

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
        ArgumentNullException.ThrowIfNull(referenceAudio);
        ArgumentNullException.ThrowIfNull(referenceText);
        ArgumentNullException.ThrowIfNull(text);
        options ??= new F5TtsOptions();

        var genText = options.TextNormalizer is null ? text : options.TextNormalizer(text);
        var tokenizer = options.Tokenizer ?? _defaultTokenizer;

        // A trailing space is needed when the reference text ends on a single-byte character,
        // otherwise the last reference word fuses with the first generated one.
        var refPart = referenceText;
        if (refPart.Length > 0 && Encoding.UTF8.GetByteCount(refPart[^1].ToString()) == 1 && !refPart.EndsWith(' '))
        {
            refPart += " ";
        }

        var textIds = tokenizer.Encode(refPart + genText);

        var refTextLen = Math.Max(1, Encoding.UTF8.GetByteCount(refPart));
        var genTextLen = Encoding.UTF8.GetByteCount(genText);
        var refAudioLen = referenceAudio.Length / HopLength + 1;
        var speed = options.Speed <= 0 ? 1.0 : options.Speed;
        var maxDuration = refAudioLen + (long)((double)refAudioLen / refTextLen * genTextLen / speed);

        lock (_sync)
        {
            var samples = RunPipeline(referenceAudio, textIds, maxDuration, options.NfeSteps);
            return new F5TtsResult(samples, SampleRate);
        }
    }

    private short[] RunPipeline(short[] referenceAudio, int[] textIds, long maxDuration, int nfeSteps)
    {
        var audioTensor = new DenseTensor<short>(referenceAudio, [1, 1, referenceAudio.Length]);
        var textTensor = new DenseTensor<int>(textIds, [1, textIds.Length]);
        var maxDurTensor = new DenseTensor<long>(new long[] { maxDuration }, new int[] { 1 });

        DenseTensor<float> noise, ropeCosQ, ropeSinQ, ropeCosK, ropeSinK, catMelText, catMelTextDrop;
        DenseTensor<long> refSignalLen;

        using (var pre = _preprocess.Run([
            NamedOnnxValue.CreateFromTensor("audio", audioTensor),
            NamedOnnxValue.CreateFromTensor("text_ids", textTensor),
            NamedOnnxValue.CreateFromTensor("max_duration", maxDurTensor),
        ]))
        {
            noise = Copy<float>(pre, "noise");
            ropeCosQ = Copy<float>(pre, "rope_cos_q");
            ropeSinQ = Copy<float>(pre, "rope_sin_q");
            ropeCosK = Copy<float>(pre, "rope_cos_k");
            ropeSinK = Copy<float>(pre, "rope_sin_k");
            catMelText = Copy<float>(pre, "cat_mel_text");
            catMelTextDrop = Copy<float>(pre, "cat_mel_text_drop");
            refSignalLen = Copy<long>(pre, "ref_signal_len");
        }

        var x = noise;
        for (var step = 0; step < nfeSteps; step++)
        {
            var timeStep = new DenseTensor<int>(new int[] { step }, new int[] { 1 });
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
            x = Copy<float>(res, "denoised");
        }

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

    public void Dispose()
    {
        _preprocess.Dispose();
        _transformer.Dispose();
        _decode.Dispose();
    }
}
