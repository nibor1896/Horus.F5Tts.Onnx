using Horus.F5Tts.Onnx;
using Xunit;

namespace Horus.F5Tts.Onnx.Tests;

/// <summary>Loads a real model set once for the model-backed tests, if one is available.
///
/// These tests are opt-in: point <c>F5_TEST_MODELS</c> at a folder containing
/// <c>F5_Preprocess.onnx</c>, <c>F5_Transformer.onnx</c>, <c>F5_Decode.onnx</c> and <c>vocab.txt</c>.
/// Without it they skip, so CI stays fast and does not need the ~1.4 GB download (whose weights are
/// non-commercially licensed anyway).</summary>
public sealed class ModelFixture : IDisposable
{
    private F5TtsModel? _model;

    /// <summary>The model folder, or null when the tests should skip.</summary>
    public static string? ModelDir { get; } = Resolve();

    /// <summary>The loaded model. Lazy: loading is heavy, and the fixture is constructed even when
    /// every test in the class ends up skipping.</summary>
    public F5TtsModel Model => _model ??= F5TtsModel.Load(
        Path.Combine(ModelDir!, "F5_Preprocess.onnx"),
        Path.Combine(ModelDir!, "F5_Transformer.onnx"),
        Path.Combine(ModelDir!, "F5_Decode.onnx"),
        Path.Combine(ModelDir!, "vocab.txt"));

    private static string? Resolve()
    {
        var dir = Environment.GetEnvironmentVariable("F5_TEST_MODELS");
        if (string.IsNullOrWhiteSpace(dir) || !Directory.Exists(dir))
        {
            return null;
        }

        string[] required = ["F5_Preprocess.onnx", "F5_Transformer.onnx", "F5_Decode.onnx", "vocab.txt"];
        return required.All(f => File.Exists(Path.Combine(dir, f))) ? dir : null;
    }

    public void Dispose() => _model?.Dispose();
}

/// <summary>A <see cref="FactAttribute"/> that skips itself unless a real model set is configured.</summary>
public sealed class ModelFactAttribute : FactAttribute
{
    public ModelFactAttribute()
    {
        if (ModelFixture.ModelDir is null)
        {
            Skip = "Set F5_TEST_MODELS to a folder containing F5_Preprocess.onnx, F5_Transformer.onnx, " +
                   "F5_Decode.onnx and vocab.txt to run the model-backed tests.";
        }
    }
}
