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
    // numeric dates
    [InlineData("3.8.2026", "dritter August zweitausendsechsundzwanzig")]
    [InlineData("am 3.8.2026", "am dritten August zweitausendsechsundzwanzig")]
    // day + month name
    [InlineData("3. August", "dritter August")]
    [InlineData("am 3. August", "am dritten August")]
    [InlineData("der 3. August", "der dritte August")]
    [InlineData("21. Dezember", "einundzwanzigster Dezember")]
    [InlineData("1. Januar 2026", "erster Januar zweitausendsechsundzwanzig")]
    // article-governed ordinals (non-date)
    [InlineData("der 3. Platz", "der dritte Platz")]
    [InlineData("am 5. Tag", "am fünften Tag")]
    [InlineData("die 2. Runde", "die zweite Runde")]
    [InlineData("beim 8. Versuch", "beim achten Versuch")]
    // times
    [InlineData("14:30", "vierzehn Uhr dreißig")]
    [InlineData("14:00", "vierzehn Uhr")]
    [InlineData("9:05", "neun Uhr fünf")]
    [InlineData("um 14:30 Uhr", "um vierzehn Uhr dreißig")]     // no doubled "Uhr"
    // deferred / must NOT be guessed as an ordinal: sentence-final number after a non-article word
    [InlineData("Ich zählte bis 5. Dann kam Ruhe.", "Ich zählte bis fünf. Dann kam Ruhe.")]
    public void Normalize_handles_dates_times_and_ordinals(string input, string expected)
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
