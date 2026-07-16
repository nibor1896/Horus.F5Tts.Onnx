using Horus.F5Tts.Onnx;
using Xunit;

namespace Horus.F5Tts.Onnx.Tests;

/// <summary>Tests that drive the real ONNX pipeline. Opt-in via <c>F5_TEST_MODELS</c> — see
/// <see cref="ModelFixture"/>.
///
/// These check the <b>mechanics</b>: that async is just the sync path on another thread, that
/// cancellation is honoured, that a seed really does reproduce audio end-to-end. They deliberately
/// do <b>not</b> judge audio quality — no assertion can hear "does this sound like speech?", and
/// pretending otherwise is how a fluent-but-wrong output slips through. Quality stays a listening
/// test plus a Whisper transcription.
///
/// A 1 s synthetic reference and a low NFE keep each test at seconds instead of ~35 s; none of the
/// properties under test depend on the audio being good.</summary>
public class ModelBackedTests : IClassFixture<ModelFixture>
{
    private readonly ModelFixture _fixture;

    public ModelBackedTests(ModelFixture fixture) => _fixture = fixture;

    private const string RefText = "This is the reference clip.";
    private const string GenText = "Hello there.";

    private static F5TtsOptions Fast(int? seed = null) => new() { NfeSteps = 4, Seed = seed };

    private static short[] Reference()
    {
        var samples = new short[24000]; // 1 s; content is irrelevant to what these tests assert
        var rnd = new Random(7);
        for (var i = 0; i < samples.Length; i++)
        {
            samples[i] = (short)rnd.Next(-3000, 3000);
        }

        return samples;
    }

    [ModelFact]
    public async Task SynthesizeAsync_returns_audio_at_the_model_rate()
    {
        var result = await _fixture.Model.SynthesizeAsync(Reference(), RefText, GenText, Fast());

        Assert.Equal(24000, result.SampleRate);
        Assert.NotEmpty(result.Samples);
        Assert.True(result.DurationSeconds > 0);
        Assert.Contains(result.Samples, s => s != 0); // not digital silence
    }

    [ModelFact]
    public async Task SynthesizeAsync_throws_when_the_token_is_already_cancelled()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => _fixture.Model.SynthesizeAsync(Reference(), RefText, GenText, Fast(), cts.Token));
    }

    [ModelFact]
    public async Task SynthesizeAsync_does_not_run_a_long_synthesis_to_completion_when_cancelled()
    {
        // Enough steps that there is something to abandon. Whether the token is caught before the
        // first step or between two of them is timing-dependent and both are correct — the point is
        // that the call does not ignore cancellation and grind to the end.
        var manySteps = new F5TtsOptions { NfeSteps = 32 };
        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(50));

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => _fixture.Model.SynthesizeAsync(Reference(), RefText, GenText, manySteps, cts.Token));
    }

    [ModelFact]
    public async Task A_queued_request_can_be_cancelled_while_waiting_for_its_turn()
    {
        // The whole point of serializing with a semaphore instead of a lock. The second call is stuck
        // behind the first; cancelling it must take effect straight away rather than after the first
        // one finally finishes. With a lock the waiting thread could not observe the token at all.
        var manySteps = new F5TtsOptions { NfeSteps = 32 };
        using var blockerCts = new CancellationTokenSource();
        using var queuedCts = new CancellationTokenSource();

        var blocker = _fixture.Model.SynthesizeAsync(Reference(), RefText, GenText, manySteps, blockerCts.Token);
        await Task.Delay(250); // give the blocker time to take the gate
        var queued = _fixture.Model.SynthesizeAsync(Reference(), RefText, GenText, manySteps, queuedCts.Token);

        queuedCts.Cancel();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => queued);
        sw.Stop();

        // Generous bound: the blocker needs many seconds, so returning this quickly proves the queued
        // call did not sit and wait for it.
        Assert.True(sw.ElapsedMilliseconds < 5000,
            $"cancelling a queued request took {sw.ElapsedMilliseconds} ms — it waited for the one in front");

        blockerCts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => blocker);
    }

    [ModelFact]
    public void The_same_seed_reproduces_identical_audio()
    {
        // The end-to-end version of the promise F5TtsOptions.Seed makes in its own docs. The unit
        // tests pin the noise generator in isolation; this pins the whole pipeline.
        var first = _fixture.Model.Synthesize(Reference(), RefText, GenText, Fast(12345));
        var second = _fixture.Model.Synthesize(Reference(), RefText, GenText, Fast(12345));

        Assert.Equal(first.Samples, second.Samples);
    }

    [ModelFact]
    public void Different_seeds_produce_different_audio()
    {
        var first = _fixture.Model.Synthesize(Reference(), RefText, GenText, Fast(1));
        var second = _fixture.Model.Synthesize(Reference(), RefText, GenText, Fast(2));

        Assert.NotEqual(first.Samples, second.Samples);
    }

    [ModelFact]
    public void Without_a_seed_the_audio_varies_between_runs()
    {
        // The documented default: null means fresh Gaussian noise per call, so timbre varies.
        var first = _fixture.Model.Synthesize(Reference(), RefText, GenText, Fast());
        var second = _fixture.Model.Synthesize(Reference(), RefText, GenText, Fast());

        Assert.NotEqual(first.Samples, second.Samples);
    }

    [ModelFact]
    public async Task Async_and_sync_produce_identical_audio_for_the_same_seed()
    {
        // SynthesizeAsync must be nothing more than Synthesize on a background thread. This is the
        // regression guard for the refactor that introduced it.
        var sync = _fixture.Model.Synthesize(Reference(), RefText, GenText, Fast(999));
        var async = await _fixture.Model.SynthesizeAsync(Reference(), RefText, GenText, Fast(999));

        Assert.Equal(sync.Samples, async.Samples);
    }
}
