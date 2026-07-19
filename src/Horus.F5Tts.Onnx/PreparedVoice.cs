namespace Horus.F5Tts.Onnx;

/// <summary>A reference voice bound to a model, so you pass the clip and its transcript <b>once</b>
/// and then synthesize with just the text — handy when one voice speaks many things (an assistant, a
/// chat UI). Create it with <see cref="F5TtsModel.PrepareVoice"/> or
/// <see cref="F5TtsModel.PrepareVoiceFromWav"/>.
///
/// <para>This is an <b>ergonomic</b> convenience, not a speed optimisation: it caches nothing about
/// the model's work. F5-TTS bakes the reference processing into the same graph pass as the generated
/// text (it cannot be reused across different texts), and that step is ~0.3 % of a synthesis anyway —
/// the transformer loop is the cost. Each call does the full pipeline, identical to calling the model
/// directly with the same reference.</para></summary>
public sealed class PreparedVoice
{
    private readonly F5TtsModel _model;
    private readonly short[] _referenceAudio;
    private readonly string _referenceText;

    internal PreparedVoice(F5TtsModel model, short[] referenceAudio, string referenceText)
    {
        _model = model;
        _referenceAudio = referenceAudio;
        _referenceText = referenceText;
    }

    /// <inheritdoc cref="F5TtsModel.Synthesize"/>
    public F5TtsResult Synthesize(string text, F5TtsOptions? options = null)
        => _model.Synthesize(_referenceAudio, _referenceText, text, options);

    /// <inheritdoc cref="F5TtsModel.SynthesizeAsync"/>
    public Task<F5TtsResult> SynthesizeAsync(string text, F5TtsOptions? options = null,
        CancellationToken cancellationToken = default)
        => _model.SynthesizeAsync(_referenceAudio, _referenceText, text, options, cancellationToken);

    /// <inheritdoc cref="F5TtsModel.SynthesizeLong"/>
    public F5TtsResult SynthesizeLong(string text, F5TtsOptions? options = null,
        double crossFadeSeconds = F5TtsModel.DefaultCrossFadeSeconds)
        => _model.SynthesizeLong(_referenceAudio, _referenceText, text, options, crossFadeSeconds);

    /// <inheritdoc cref="F5TtsModel.SynthesizeLongAsync"/>
    public Task<F5TtsResult> SynthesizeLongAsync(string text, F5TtsOptions? options = null,
        double crossFadeSeconds = F5TtsModel.DefaultCrossFadeSeconds, CancellationToken cancellationToken = default)
        => _model.SynthesizeLongAsync(_referenceAudio, _referenceText, text, options, crossFadeSeconds, cancellationToken);

    /// <inheritdoc cref="F5TtsModel.SynthesizeStreamAsync"/>
    public IAsyncEnumerable<F5TtsChunk> SynthesizeStreamAsync(string text, F5TtsOptions? options = null,
        double crossFadeSeconds = F5TtsModel.DefaultCrossFadeSeconds, CancellationToken cancellationToken = default)
        => _model.SynthesizeStreamAsync(_referenceAudio, _referenceText, text, options, crossFadeSeconds, cancellationToken);
}
