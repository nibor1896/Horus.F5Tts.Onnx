using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace Horus.F5Tts.Onnx;

/// <summary>Spells out what a German F5-TTS checkpoint cannot pronounce — numbers, currency, percent,
/// common abbreviations, a few symbols — into plain words. Assign it to
/// <see cref="F5TtsOptions.TextNormalizer"/>:
///
/// <code>options.TextNormalizer = GermanTextNormalizer.Normalize;</code>
///
/// It is deterministic and stateless, and it only rewrites patterns it recognises — ordinary prose
/// passes through untouched. Context- or inflection-dependent cases (bare ordinals like "3.", dates,
/// clock times) are deliberately left alone: a wrong reading is worse than a skipped symbol, and
/// those need sentence context this pass does not have.</summary>
public static partial class GermanTextNormalizer
{
    // 0..19 as words; index is the value. "eins" is the standalone form of 1 — compounds use "ein".
    private static readonly string[] Ones =
    [
        "null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun", "zehn",
        "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn", "sechzehn", "siebzehn", "achtzehn", "neunzehn",
    ];

    // Tens; index is the tens digit (2..9). Note sech-/sieb- lose their full stem here.
    private static readonly string[] Tens =
        ["", "", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig", "achtzig", "neunzig"];

    private static readonly (string Abbr, string Full)[] Abbreviations =
    [
        // Multi-dot forms first so a shorter one never eats part of a longer one.
        ("z.B.", "zum Beispiel"), ("z.T.", "zum Teil"), ("d.h.", "das heißt"), ("u.a.", "unter anderem"),
        ("i.d.R.", "in der Regel"), ("u.U.", "unter Umständen"), ("o.ä.", "oder ähnliches"),
        ("u.Ä.", "und Ähnliches"), ("usw.", "und so weiter"), ("etc.", "et cetera"),
        ("bzw.", "beziehungsweise"), ("ca.", "circa"), ("ggf.", "gegebenenfalls"), ("evtl.", "eventuell"),
        ("inkl.", "inklusive"), ("exkl.", "exklusive"), ("vs.", "versus"), ("Nr.", "Nummer"),
        ("Dr.", "Doktor"), ("Prof.", "Professor"), ("Str.", "Straße"), ("Tel.", "Telefon"),
        ("Mio.", "Millionen"), ("Mrd.", "Milliarden"),
    ];

    /// <summary>Rewrites numbers, currency, percent, abbreviations and a few symbols in
    /// <paramref name="text"/> into spoken German words. Returns the text unchanged where nothing
    /// matched.</summary>
    public static string Normalize(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        text = ExpandAbbreviations(text);
        text = CurrencyRegex().Replace(text, CurrencyEval);
        text = PercentRegex().Replace(text, m => SpellDecimalOrInt(m.Groups[1].Value) + " Prozent");
        text = TemperatureRegex().Replace(text, m => SpellDecimalOrInt(m.Groups[1].Value) + " Grad Celsius");
        text = DegreeRegex().Replace(text, m => SpellDecimalOrInt(m.Groups[1].Value) + " Grad");
        text = MinusRegex().Replace(text, "${pre}minus ");        // a leading "-" before digits
        text = ThousandsRegex().Replace(text, m => SpellInteger(m.Value.Replace(".", "")));
        text = DecimalRegex().Replace(text, m => SpellDecimal(m.Groups[1].Value, m.Groups[2].Value));
        text = IntegerRegex().Replace(text, m => SpellInteger(m.Value));
        text = ExpandSymbols(text);
        return text;
    }

    private static string ExpandAbbreviations(string text)
    {
        foreach (var (abbr, full) in Abbreviations)
        {
            // Bounded so "Dr." matches but "Dr" inside a word does not, and the trailing dot is
            // consumed. Case-sensitive on purpose: "max." → maximal, but the name "Max." is left alone.
            var pattern = $@"(?<![\w.]){Regex.Escape(abbr)}(?!\w)";
            text = Regex.Replace(text, pattern, full);
        }

        return text;
    }

    private static string ExpandSymbols(string text)
    {
        text = Regex.Replace(text, @"(?<=\s)&(?=\s)", "und");
        text = Regex.Replace(text, @"(?<=\s)\+(?=\s)", "plus");
        text = Regex.Replace(text, @"(?<=\s)=(?=\s)", "gleich");
        text = Regex.Replace(text, @"§", "Paragraph");
        return text;
    }

    private static string CurrencyEval(Match m)
    {
        // amount + optional ,cc + € (German order), or $ + amount.
        var whole = (m.Groups["euro"].Success ? m.Groups["euro"].Value : m.Groups["dol"].Value).Replace(".", "");
        var cents = m.Groups["cents"].Value;
        var unit = m.Groups["dol"].Success ? "Dollar" : "Euro";
        var amount = whole == "1" ? "ein" : SpellInteger(whole);   // "ein Euro", never "eins Euro"
        if (string.IsNullOrEmpty(cents) || cents == "00")
        {
            return $"{amount} {unit}";
        }

        return $"{amount} {unit} {SpellInteger(cents.TrimStart('0') is { Length: > 0 } c ? c : "0")}";
    }

    private static string SpellDecimalOrInt(string number) =>
        number.Contains(',') ? SpellDecimal(number[..number.IndexOf(',')], number[(number.IndexOf(',') + 1)..])
            : SpellInteger(number.Replace(".", ""));

    private static string SpellDecimal(string intPart, string fracPart)
    {
        var sb = new StringBuilder(SpellInteger(intPart.Replace(".", "")));
        sb.Append(" Komma");
        foreach (var d in fracPart)                                // fractional digits are read one by one
        {
            sb.Append(' ').Append(Ones[d - '0']);
        }

        return sb.ToString();
    }

    /// <summary>Spells a non-negative integer given as a digit string (up to the billions). Out of
    /// range or unparsable input is returned as-is rather than guessed.</summary>
    internal static string SpellInteger(string digits)
    {
        if (!long.TryParse(digits, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out var n))
        {
            return digits;
        }

        return SpellCardinal(n);
    }

    /// <summary>The German cardinal for <paramref name="n"/> (supports negatives and up to &lt; 10^12).
    /// Numbers below a million are one word; millions and billions are separated with spaces, as
    /// German writes them.</summary>
    internal static string SpellCardinal(long n)
    {
        if (n == 0)
        {
            return "null";
        }

        if (n < 0)
        {
            return "minus " + SpellCardinal(-n);
        }

        if (n >= 1_000_000_000_000L)
        {
            return n.ToString(CultureInfo.InvariantCulture);       // beyond our range: leave the digits
        }

        var parts = new List<string>();
        var billions = n / 1_000_000_000;
        var millions = n / 1_000_000 % 1000;
        var belowMillion = n % 1_000_000;

        if (billions > 0)
        {
            parts.Add(billions == 1 ? "eine Milliarde" : BelowThousand((int)billions) + " Milliarden");
        }

        if (millions > 0)
        {
            parts.Add(millions == 1 ? "eine Million" : BelowThousand((int)millions) + " Millionen");
        }

        if (belowMillion > 0)
        {
            parts.Add(BelowMillion((int)belowMillion));
        }

        return string.Join(" ", parts);
    }

    private static string BelowMillion(int n)                      // 1 .. 999_999, one word
    {
        var thousands = n / 1000;
        var rest = n % 1000;
        if (thousands == 0)
        {
            return BelowThousand(rest);
        }

        var word = (thousands == 1 ? "ein" : BelowThousand(thousands)) + "tausend";
        return rest == 0 ? word : word + BelowThousand(rest);
    }

    private static string BelowThousand(int n)                     // 1 .. 999, one word
    {
        var hundreds = n / 100;
        var rest = n % 100;
        if (hundreds == 0)
        {
            return BelowHundred(rest);
        }

        var word = (hundreds == 1 ? "ein" : Ones[hundreds]) + "hundert";
        return rest == 0 ? word : word + BelowHundred(rest);
    }

    private static string BelowHundred(int n)                      // 0 .. 99
    {
        if (n < 20)
        {
            return Ones[n];
        }

        var unit = n % 10;
        var tens = Tens[n / 10];
        if (unit == 0)
        {
            return tens;
        }

        return (unit == 1 ? "ein" : Ones[unit]) + "und" + tens;    // einundzwanzig, fünfundvierzig
    }

    [GeneratedRegex(@"(?:(?<euro>\d{1,3}(?:\.\d{3})*|\d+)(?:,(?<cents>\d{2}))?\s*€|\$\s*(?<dol>\d{1,3}(?:\.\d{3})*|\d+))")]
    private static partial Regex CurrencyRegex();

    [GeneratedRegex(@"(\d{1,3}(?:\.\d{3})*(?:,\d+)?|\d+(?:,\d+)?)\s*%")]
    private static partial Regex PercentRegex();

    [GeneratedRegex(@"(-?\d+(?:,\d+)?)\s*°\s?C\b")]
    private static partial Regex TemperatureRegex();

    [GeneratedRegex(@"(-?\d+(?:,\d+)?)\s*°(?!\s?C)")]
    private static partial Regex DegreeRegex();

    [GeneratedRegex(@"(?<pre>^|\s|\()-(?=\d)")]
    private static partial Regex MinusRegex();

    [GeneratedRegex(@"(\d+),(\d+)")]
    private static partial Regex DecimalRegex();

    [GeneratedRegex(@"\d{1,3}(?:\.\d{3})+")]
    private static partial Regex ThousandsRegex();

    [GeneratedRegex(@"\d+")]
    private static partial Regex IntegerRegex();
}
