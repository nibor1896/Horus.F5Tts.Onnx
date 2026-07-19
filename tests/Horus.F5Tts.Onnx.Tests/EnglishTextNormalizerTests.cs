using Horus.F5Tts.Onnx;
using Xunit;

namespace Horus.F5Tts.Onnx.Tests;

public class EnglishTextNormalizerTests
{
    [Theory]
    [InlineData(0, "zero")]
    [InlineData(1, "one")]
    [InlineData(13, "thirteen")]
    [InlineData(20, "twenty")]
    [InlineData(21, "twenty-one")]
    [InlineData(45, "forty-five")]
    [InlineData(100, "one hundred")]
    [InlineData(123, "one hundred twenty-three")]
    [InlineData(1000, "one thousand")]
    [InlineData(2026, "two thousand twenty-six")]
    [InlineData(1234567, "one million two hundred thirty-four thousand five hundred sixty-seven")]
    [InlineData(1000000000, "one billion")]
    [InlineData(-5, "minus five")]
    public void SpellCardinal_matches_written_English(long n, string expected)
    {
        Assert.Equal(expected, EnglishTextNormalizer.SpellCardinal(n));
    }

    [Theory]
    [InlineData(1, "first")]
    [InlineData(2, "second")]
    [InlineData(3, "third")]
    [InlineData(5, "fifth")]
    [InlineData(8, "eighth")]
    [InlineData(12, "twelfth")]
    [InlineData(20, "twentieth")]
    [InlineData(21, "twenty-first")]
    [InlineData(43, "forty-third")]
    [InlineData(100, "one hundredth")]
    [InlineData(101, "one hundred first")]
    [InlineData(121, "one hundred twenty-first")]
    public void SpellOrdinal_matches_written_English(int n, string expected)
    {
        Assert.Equal(expected, EnglishTextNormalizer.SpellOrdinal(n));
    }

    [Theory]
    // percent
    [InlineData("50%", "fifty percent")]
    [InlineData("3.5 %", "three point five percent")]
    // currency ($ before, English . decimal / , thousands)
    [InlineData("$5", "five dollars")]
    [InlineData("$1", "one dollar")]
    [InlineData("$3.50", "three dollars and fifty cents")]
    [InlineData("$1,000", "one thousand dollars")]
    [InlineData("£5", "five pounds")]
    // decimals & thousands
    [InlineData("3.14", "three point one four")]
    [InlineData("1,000", "one thousand")]
    // ordinals (written explicitly in English)
    [InlineData("the 1st place", "the first place")]
    [InlineData("August 3rd", "August third")]
    [InlineData("the 21st century", "the twenty-first century")]
    // times
    [InlineData("2:30", "two thirty")]
    [InlineData("2:00", "two o'clock")]
    [InlineData("2:05", "two oh five")]
    [InlineData("2:30 pm", "two thirty PM")]
    [InlineData("14:30", "fourteen thirty")]
    // abbreviations
    [InlineData("e.g. cats", "for example cats")]
    [InlineData("Dr. Smith", "Doctor Smith")]
    [InlineData("No. 5", "Number five")]
    // symbols & plain integers
    [InlineData("A & B", "A and B")]
    [InlineData("I have 21 cats.", "I have twenty-one cats.")]
    [InlineData("It is -5 outside.", "It is minus five outside.")]
    [InlineData("20 °C", "twenty degrees Celsius")]
    // the README example, locked so the docs cannot drift
    [InlineData("I saved $1,000 (50%) by the 3rd, at 2:30 pm",
        "I saved one thousand dollars (fifty percent) by the third, at two thirty PM")]
    public void Normalize_rewrites_the_pattern(string input, string expected)
    {
        Assert.Equal(expected, EnglishTextNormalizer.Normalize(input));
    }

    [Theory]
    [InlineData("The quick brown fox jumps over the lazy dog.")]
    [InlineData("Hello, how are you today?")]
    public void Normalize_leaves_plain_prose_untouched(string prose)
    {
        Assert.Equal(prose, EnglishTextNormalizer.Normalize(prose));
    }

    [Fact]
    public void Normalize_is_idempotent()
    {
        var once = EnglishTextNormalizer.Normalize("I saved 50% of $1,000, e.g. on the 3rd.");
        Assert.Equal(once, EnglishTextNormalizer.Normalize(once));
    }
}
