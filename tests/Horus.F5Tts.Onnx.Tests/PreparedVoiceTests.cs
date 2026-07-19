using Horus.F5Tts.Onnx;
using Xunit;

namespace Horus.F5Tts.Onnx.Tests;

/// <summary>PreparedVoice is a thin binding over the model — these confirm it delegates to exactly the
/// same call, i.e. binding the reference once produces byte-identical audio to passing it each time.</summary>
public class PreparedVoiceTests : IClassFixture<ModelFixture>
{
    private readonly ModelFixture _fixture;

    public PreparedVoiceTests(ModelFixture fixture) => _fixture = fixture;

    private const string RefText = "This is the reference clip.";
    private const string GenText = "Hello there.";

    private static F5TtsOptions Fast(int? seed = null) => new() { NfeSteps = 4, Seed = seed };

    private static short[] Reference()
    {
        var samples = new short[24000];
        var rnd = new Random(7);
        for (var i = 0; i < samples.Length; i++)
        {
            samples[i] = (short)rnd.Next(-3000, 3000);
        }

        return samples;
    }

    [ModelFact]
    public void PreparedVoice_Synthesize_matches_calling_the_model_directly()
    {
        var reference = Reference();
        var direct = _fixture.Model.Synthesize(reference, RefText, GenText, Fast(42));

        var voice = _fixture.Model.PrepareVoice(reference, RefText);
        var viaVoice = voice.Synthesize(GenText, Fast(42));

        Assert.Equal(direct.Samples, viaVoice.Samples);
    }

    [ModelFact]
    public void PrepareVoiceFromWav_loads_the_clip_and_delegates_the_same()
    {
        // Write a reference clip to a real .wav, then check loading it through PrepareVoiceFromWav
        // gives the same audio as loading it by hand and calling the model.
        var path = Path.Combine(Path.GetTempPath(), $"prepvoice_{Guid.NewGuid():N}.wav");
        File.WriteAllBytes(path, WavAudio.WritePcm16(Reference(), _fixture.Model.OutputSampleRate));
        try
        {
            var expected = _fixture.Model.Synthesize(
                WavAudio.ReadPcm16Resampled(path, _fixture.Model.OutputSampleRate), RefText, GenText, Fast(9));

            var voice = _fixture.Model.PrepareVoiceFromWav(path, RefText);
            var actual = voice.Synthesize(GenText, Fast(9));

            Assert.Equal(expected.Samples, actual.Samples);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [ModelFact]
    public void PrepareVoice_rejects_null_arguments()
    {
        Assert.Throws<ArgumentNullException>(() => _fixture.Model.PrepareVoice(null!, RefText));
        Assert.Throws<ArgumentNullException>(() => _fixture.Model.PrepareVoice(Reference(), null!));
        Assert.Throws<ArgumentNullException>(() => _fixture.Model.PrepareVoiceFromWav(null!, RefText));
    }
}
