using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using Horus.F5Tts.Onnx;

// Minimal end-to-end demo: load a model set, synthesize one sentence, write a .wav.
//
// The sample is language-agnostic — point <modelDir> at any F5-TTS ONNX export. The language and
// accent come from that checkpoint; the speaking voice comes from <reference.wav>.
//
// Usage:
//   Horus.F5Tts.Onnx.Sample <modelDir> <reference.wav> "<reference text>" "<text to speak>" <out.wav> [seed] [stream]
//
// Pass a [seed] to make the output reproducible: the same reference, text and seed always produce
// identical audio. The sample then prints a fingerprint of the samples, which turns "did my change
// alter the output?" into a yes/no comparison instead of a listening test. Add "stream" to synthesize
// chunk-by-chunk (SynthesizeStreamAsync) — the audio, and so the fingerprint, is identical to the
// non-streaming run; only the first-audio latency differs.
//
// Examples:
//   # German checkpoint (huggingface.co/nibor1896/F5-TTS-German-ONNX)
//   Horus.F5Tts.Onnx.Sample models/de ref_de.wav "Der Referenztext." "Hallo, wie geht es dir?" out.wav
//   # English checkpoint (a base F5-TTS ONNX export)
//   Horus.F5Tts.Onnx.Sample models/en ref_en.wav "The reference transcript." "Hello, how are you?" out.wav
//
// <modelDir> must contain F5_Preprocess.onnx, F5_Transformer.onnx, F5_Decode.onnx and vocab.txt.
// <reference.wav> is any 16-bit PCM WAV — the sample resamples/down-mixes it as needed.

if (args.Length < 5)
{
    Console.Error.WriteLine(
        "Usage: <modelDir> <reference.wav> \"<reference text>\" \"<text to speak>\" <out.wav> [seed] [stream]");
    return 1;
}

var modelDir = args[0];

// Progress matters here: loading is a multi-hundred-megabyte read and synthesis is silent
// compute, so without these lines the sample looks hung for the better part of a minute.
Console.WriteLine("Loading models ...");
var swLoad = Stopwatch.StartNew();

using var model = F5TtsModel.Load(
    Path.Combine(modelDir, "F5_Preprocess.onnx"),
    Path.Combine(modelDir, "F5_Transformer.onnx"),
    Path.Combine(modelDir, "F5_Decode.onnx"),
    Path.Combine(modelDir, "vocab.txt"));
// To use a GPU instead of CPU, add the matching runtime package and pass e.g.:
//   configureSession: o => o.AppendExecutionProvider_DML(0)

swLoad.Stop();
Console.WriteLine($"  loaded in {swLoad.ElapsedMilliseconds} ms");

// Any sample rate works: the clip is converted to whatever the model expects (24 kHz).
var referenceAudio = WavAudio.ReadPcm16Resampled(args[1], model.OutputSampleRate);

// Trailing args (any order): an integer [seed] pins the output; the word "stream" synthesizes via
// SynthesizeStreamAsync (chunk-by-chunk) instead of SynthesizeLong. Streaming is the same audio,
// delivered incrementally — with a seed, the two produce an identical fingerprint.
var options = new F5TtsOptions();
var streamMode = false;
for (var i = 5; i < args.Length; i++)
{
    if (string.Equals(args[i], "stream", StringComparison.OrdinalIgnoreCase))
    {
        streamMode = true;
    }
    else if (string.Equals(args[i], "norm", StringComparison.OrdinalIgnoreCase))
    {
        options.TextNormalizer = GermanTextNormalizer.Normalize;   // spell out numbers, %, €, z.B. …
    }
    else if (int.TryParse(args[i], out var seed))
    {
        options.Seed = seed;
    }
    else
    {
        Console.Error.WriteLine($"Unrecognized argument '{args[i]}' (expected a seed, 'stream' or 'norm').");
        return 1;
    }
}

if (options.TextNormalizer is not null)
{
    Console.WriteLine($"Normalisiert -> \"{options.TextNormalizer(args[3])}\"");
}

Console.WriteLine($"Synthesizing: \"{args[3]}\"{(streamMode ? "  [streaming]" : "")}");
Console.WriteLine("  (CPU/GPU-bound and silent until done — on CPU a large model takes tens of seconds)");

// SynthesizeLong rather than Synthesize: text that fits in one pass is handed straight to Synthesize,
// so nothing changes for a short sentence — but a paragraph gets chunked and cross-faded instead of
// coming out degraded. There is no reason for a sample to demonstrate the fragile call.
var sw = Stopwatch.StartNew();
short[] samples;
if (streamMode)
{
    // Stream the pieces and concatenate them. The audio is identical to SynthesizeLong (the
    // fingerprint below proves it) — the point of streaming is that the first chunk is ready early,
    // so we report when it arrives.
    var pieces = new List<short>();
    var first = true;
    await foreach (var chunk in model.SynthesizeStreamAsync(referenceAudio, referenceText: args[2], text: args[3], options))
    {
        if (first)
        {
            Console.WriteLine($"  first audio after {sw.ElapsedMilliseconds} ms (of {chunk.Count} chunk(s))");
            first = false;
        }

        pieces.AddRange(chunk.Samples);
    }

    samples = pieces.ToArray();
}
else
{
    samples = model.SynthesizeLong(referenceAudio, referenceText: args[2], text: args[3], options).Samples;
}

sw.Stop();

File.WriteAllBytes(args[4], WavAudio.WritePcm16(samples, model.OutputSampleRate));
var durationSeconds = (double)samples.Length / model.OutputSampleRate;
Console.WriteLine(
    $"Wrote {args[4]} — {durationSeconds:F1}s of audio, synthesized in {sw.ElapsedMilliseconds} ms.");

// A fingerprint of the raw samples. With a seed this is stable across runs, machines and execution
// providers, so re-running after a code change answers "is the audio still bit-for-bit the same?"
// without anyone having to trust their ears — and a streamed run must match the non-streamed one.
var fingerprint = Convert.ToHexString(SHA256.HashData(MemoryMarshal.AsBytes<short>(samples)))[..16];
Console.WriteLine(options.Seed is null
    ? $"Audio fingerprint: {fingerprint} (no seed — expected to differ between runs; pass a seed to pin it)"
    : $"Audio fingerprint: {fingerprint} (seed {options.Seed} — must stay identical across runs)");
return 0;
