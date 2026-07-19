namespace Horus.F5Tts.Onnx;

/// <summary>One piece of a streamed synthesis (<see cref="F5TtsModel.SynthesizeStreamAsync"/>):
/// ready-to-play PCM, plus where it sits in the whole request.
///
/// Concatenating every chunk's <see cref="Samples"/> in arrival order yields <b>exactly</b> the same
/// audio as <see cref="F5TtsModel.SynthesizeLongAsync"/> for the same inputs and seed — the stream is
/// the batch result, delivered incrementally, not a second rendering. A caller typically writes one
/// WAV header (or opens one PCM sink) up front and appends <see cref="Samples"/> as chunks arrive.</summary>
/// <param name="Samples">Signed 16-bit PCM, mono, at <see cref="SampleRate"/>. Usually the audio for
/// this text segment; occasionally empty when a very short segment's audio is still being held back
/// to cross-fade with the next one.</param>
/// <param name="SampleRate">Samples per second (24000 for F5-TTS).</param>
/// <param name="Index">Zero-based position of this chunk in the request.</param>
/// <param name="Count">Total number of chunks the text was split into (1 for short text).</param>
/// <param name="Text">The text segment this chunk speaks.</param>
public sealed record F5TtsChunk(short[] Samples, int SampleRate, int Index, int Count, string Text);
