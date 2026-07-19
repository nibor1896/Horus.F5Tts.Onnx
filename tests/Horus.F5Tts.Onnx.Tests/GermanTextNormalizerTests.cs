using Horus.F5Tts.Onnx;
using Xunit;

namespace Horus.F5Tts.Onnx.Tests;

public class GermanTextNormalizerTests
{
    [Theory]
    [InlineData(0, "null")]
    [InlineData(1, "eins")]
    [InlineData(7, "sieben")]
    [InlineData(11, "elf")]
    [InlineData(13, "dreizehn")]
    [InlineData(16, "sechzehn")]      // not "sechszehn"
    [InlineData(17, "siebzehn")]      // not "siebenzehn"
    [InlineData(20, "zwanzig")]
    [InlineData(21, "einundzwanzig")] // "ein", not "eins"
    [InlineData(45, "fünfundvierzig")]
    [InlineData(30, "dreißig")]       // ß
    [InlineData(100, "einhundert")]
    [InlineData(101, "einhunderteins")]
    [InlineData(123, "einhundertdreiundzwanzig")]
    [InlineData(200, "zweihundert")]
    [InlineData(1000, "eintausend")]
    [InlineData(2026, "zweitausendsechsundzwanzig")]
    [InlineData(21000, "einundzwanzigtausend")]
    [InlineData(123456, "einhundertdreiundzwanzigtausendvierhundertsechsundfünfzig")]
    [InlineData(1000000, "eine Million")]
    [InlineData(2000000, "zwei Millionen")]
    [InlineData(1500000, "eine Million fünfhunderttausend")]
    [InlineData(1000000000, "eine Milliarde")]
    [InlineData(-5, "minus fünf")]
    public void SpellCardinal_matches_written_German(long n, string expected)
    {
        Assert.Equal(expected, GermanTextNormalizer.SpellCardinal(n));
    }

    [Theory]
    // percent
    [InlineData("50 %", "fünfzig Prozent")]
    [InlineData("50%", "fünfzig Prozent")]
    [InlineData("3,5 %", "drei Komma fünf Prozent")]
    [InlineData("100 %", "einhundert Prozent")]
    // currency
    [InlineData("10 €", "zehn Euro")]
    [InlineData("1 €", "ein Euro")]               // "ein Euro", never "eins Euro"
    [InlineData("3,50 €", "drei Euro fünfzig")]
    [InlineData("3,05 €", "drei Euro fünf")]
    [InlineData("3,00 €", "drei Euro")]
    [InlineData("$5", "fünf Dollar")]
    // decimals & thousands (German conventions)
    [InlineData("3,5", "drei Komma fünf")]
    [InlineData("3,14", "drei Komma eins vier")]  // fractional digits read singly
    [InlineData("1.000", "eintausend")]           // "." groups thousands
    [InlineData("1.000.000", "eine Million")]
    // plain integers
    [InlineData("Ich habe 21 Katzen.", "Ich habe einundzwanzig Katzen.")]
    [InlineData("Es sind -5 Grad.", "Es sind minus fünf Grad.")]
    // temperature & degree
    [InlineData("20 °C", "zwanzig Grad Celsius")]
    [InlineData("-5 °C", "minus fünf Grad Celsius")]
    [InlineData("90°", "neunzig Grad")]
    // abbreviations
    [InlineData("z.B. Katzen", "zum Beispiel Katzen")]
    [InlineData("d.h. jetzt", "das heißt jetzt")]
    [InlineData("Dr. Müller", "Doktor Müller")]
    [InlineData("Nr. 5", "Nummer fünf")]
    [InlineData("und so weiter usw.", "und so weiter und so weiter")]
    // symbols
    [InlineData("A & B", "A und B")]
    [InlineData("§ 5", "Paragraph fünf")]
    public void Normalize_rewrites_the_pattern(string input, string expected)
    {
        Assert.Equal(expected, GermanTextNormalizer.Normalize(input));
    }

    [Theory]
    [InlineData("Ich gehe heute nach Hause und koche Abendessen.")]
    [InlineData("Der Himmel ist blau, das Gras ist grün.")]
    [InlineData("Hallo, wie geht es dir?")]
    public void Normalize_leaves_plain_prose_untouched(string prose)
    {
        Assert.Equal(prose, GermanTextNormalizer.Normalize(prose));
    }

    [Fact]
    public void The_name_Max_is_not_mistaken_for_an_abbreviation()
    {
        // "max." (lowercase) would expand, but the capitalised name must not — case-sensitive matching.
        Assert.Equal("Max kommt.", GermanTextNormalizer.Normalize("Max kommt."));
    }

    [Fact]
    public void Normalize_is_idempotent_on_already_spoken_text()
    {
        var once = GermanTextNormalizer.Normalize("Ich habe 50 % und 10 € und z.B. Katzen.");
        Assert.Equal(once, GermanTextNormalizer.Normalize(once));
    }
}
