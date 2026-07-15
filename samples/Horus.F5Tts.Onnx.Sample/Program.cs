using System.Diagnostics;
using Horus.F5Tts.Onnx;

// Minimal end-to-end demo: load a model set, synthesize one sentence, write a .wav.
//
// The sample is language-agnostic — point <modelDir> at any F5-TTS ONNX export. The language and
// accent come from that checkpoint; the speaking voice comes from <reference.wav>.
//
// Usage:
//   Horus.F5Tts.Onnx.Sample <modelDir> <reference.wav> "<reference text>" "<text to speak>" <out.wav>
//
// Examples:
//   # German checkpoint (huggingface.co/nibor1896/F5-TTS-German-ONNX)
//   Horus.F5Tts.Onnx.Sample models/de ref_de.wav "Der Referenztext." "Hallo, wie geht es dir?" out.wav
//   # English checkpoint (a base F5-TTS ONNX export)
//   Horus.F5Tts.Onnx.Sample models/en ref_en.wav "The reference transcript." "Hello, how are you?" out.wav
//
// <modelDir> must contain F5_Preprocess.onnx, F5_Transformer.onnx, F5_Decode.onnx and vocab.txt.
// <reference.wav> must be 24 kHz mono 16-bit PCM.

if (args.Length < 5)
{
    Console.Error.WriteLine(
        "Usage: <modelDir> <reference.wav> \"<reference text>\" \"<text to speak>\" <out.wav>");
    return 1;
}

var modelDir = args[0];

using var model = F5TtsModel.Load(
    Path.Combine(modelDir, "F5_Preprocess.onnx"),
    Path.Combine(modelDir, "F5_Transformer.onnx"),
    Path.Combine(modelDir, "F5_Decode.onnx"),
    Path.Combine(modelDir, "vocab.txt"));
// To use a GPU instead of CPU, add the matching runtime package and pass e.g.:
//   configureSession: o => o.AppendExecutionProvider_DML(0)

var (referenceAudio, sampleRate) = WavAudio.ReadPcm16(args[1]);
if (sampleRate != model.OutputSampleRate)
{
    Console.Error.WriteLine(
        $"Warning: reference is {sampleRate} Hz, but the model expects {model.OutputSampleRate} Hz. " +
        "Resample it for best results.");
}

var sw = Stopwatch.StartNew();
var result = model.Synthesize(referenceAudio, referenceText: args[2], text: args[3]);
sw.Stop();

File.WriteAllBytes(args[4], result.ToWav());
Console.WriteLine(
    $"Wrote {args[4]} — {result.DurationSeconds:F1}s of audio, synthesized in {sw.ElapsedMilliseconds} ms.");
return 0;
