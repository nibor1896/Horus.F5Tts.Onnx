using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace Horus.F5Tts.Onnx;

/// <summary>Spells out what an English F5-TTS checkpoint cannot pronounce — numbers, currency,
/// percent, ordinals, clock times, common abbreviations, a few symbols — into plain words. Assign it
/// to <see cref="F5TtsOptions.TextNormalizer"/>:
///
/// <code>options.TextNormalizer = EnglishTextNormalizer.Normalize;</code>
///
/// Deterministic and stateless; it rewrites only patterns it recognises and leaves prose untouched.
/// Uses the English convention (<c>.</c> is the decimal point, <c>,</c> groups thousands — the
/// opposite of German). Numeric dates like <c>3/8/2026</c> are left alone on purpose: month/day order
/// is locale-ambiguous, and a wrong reading is worse than a skipped one.</summary>
public static partial class EnglishTextNormalizer
{
    private static readonly string[] Ones =
    [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    ];

    private static readonly string[] Tens =
        ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

    // Ordinals 0..19; irregular where English is (first, second, third, fifth, eighth, ninth, twelfth).
    private static readonly string[] OrdinalOnes =
    [
        "zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
        "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth",
        "seventeenth", "eighteenth", "nineteenth",
    ];

    private static readonly string[] OrdinalTens =
        ["", "", "twentieth", "thirtieth", "fortieth", "fiftieth", "sixtieth", "seventieth", "eightieth", "ninetieth"];

    private static readonly (string Abbr, string Full)[] Abbreviations =
    [
        ("e.g.", "for example"), ("i.e.", "that is"), ("etc.", "et cetera"), ("vs.", "versus"),
        ("approx.", "approximately"), ("Mr.", "Mister"), ("Mrs.", "Missus"), ("Ms.", "Miz"),
        ("Dr.", "Doctor"), ("Prof.", "Professor"), ("No.", "Number"), ("a.m.", "AM"), ("p.m.", "PM"),
    ];

    /// <summary>Rewrites numbers, currency, percent, ordinals, clock times, abbreviations and a few
    /// symbols in <paramref name="text"/> into spoken English words.</summary>
    public static string Normalize(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        text = ExpandAbbreviations(text);
        text = CurrencyRegex().Replace(text, CurrencyEval);
        text = PercentRegex().Replace(text, m => SpellDecimalOrInt(m.Groups[1].Value) + " percent");
        text = TemperatureRegex().Replace(text, m => SpellDecimalOrInt(m.Groups[1].Value) + " degrees Celsius");
        text = DegreeRegex().Replace(text, m => SpellDecimalOrInt(m.Groups[1].Value) + " degrees");
        text = TimeRegex().Replace(text, TimeEval);                    // 2:30, 2:30 pm
        text = OrdinalRegex().Replace(text, m => SpellOrdinal(int.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture)));
        text = MinusRegex().Replace(text, "${pre}minus ");
        text = ThousandsRegex().Replace(text, m => SpellInteger(m.Value.Replace(",", "")));
        text = DecimalRegex().Replace(text, m => SpellDecimal(m.Groups[1].Value, m.Groups[2].Value));
        text = IntegerRegex().Replace(text, m => SpellInteger(m.Value));
        text = ExpandSymbols(text);
        return text;
    }

    private static string ExpandAbbreviations(string text)
    {
        foreach (var (abbr, full) in Abbreviations)
        {
            text = Regex.Replace(text, $@"(?<![\w.]){Regex.Escape(abbr)}(?!\w)", full);
        }

        return text;
    }

    private static string ExpandSymbols(string text)
    {
        text = Regex.Replace(text, @"(?<=\s)&(?=\s)", "and");
        text = Regex.Replace(text, @"(?<=\s)\+(?=\s)", "plus");
        text = Regex.Replace(text, @"(?<=\s)=(?=\s)", "equals");
        return text;
    }

    private static string CurrencyEval(Match m)
    {
        var whole = m.Groups["amt"].Value.Replace(",", "");
        var cents = m.Groups["cents"].Value;
        var (unit, unit1) = m.Groups["sym"].Value switch
        {
            "£" => ("pounds", "pound"),
            "€" => ("euros", "euro"),
            _ => ("dollars", "dollar"),
        };
        var dollars = whole == "1" ? $"one {unit1}" : $"{SpellInteger(whole)} {unit}";
        if (string.IsNullOrEmpty(cents) || cents == "00")
        {
            return dollars;
        }

        var centVal = int.Parse(cents, CultureInfo.InvariantCulture);
        return $"{dollars} and {SpellCardinal(centVal)} cent{(centVal == 1 ? "" : "s")}";
    }

    private static string TimeEval(Match m)
    {
        var hour = int.Parse(m.Groups["h"].Value, CultureInfo.InvariantCulture);
        var minute = int.Parse(m.Groups["m"].Value, CultureInfo.InvariantCulture);
        if (hour > 23 || minute > 59)
        {
            return m.Value;
        }

        var suffix = m.Groups["ap"].Success ? " " + m.Groups["ap"].Value.Replace(".", "").ToUpperInvariant() : "";
        var spoken = minute == 0 ? $"{SpellCardinal(hour)} o'clock"
            : minute < 10 ? $"{SpellCardinal(hour)} oh {SpellCardinal(minute)}"
            : $"{SpellCardinal(hour)} {SpellCardinal(minute)}";
        return spoken + suffix;
    }

    private static string SpellDecimalOrInt(string number) =>
        number.Contains('.') ? SpellDecimal(number[..number.IndexOf('.')], number[(number.IndexOf('.') + 1)..])
            : SpellInteger(number.Replace(",", ""));

    private static string SpellDecimal(string intPart, string fracPart)
    {
        var sb = new StringBuilder(SpellInteger(intPart.Replace(",", "")));
        sb.Append(" point");
        foreach (var d in fracPart)
        {
            sb.Append(' ').Append(Ones[d - '0']);
        }

        return sb.ToString();
    }

    internal static string SpellInteger(string digits) =>
        long.TryParse(digits, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out var n)
            ? SpellCardinal(n)
            : digits;

    /// <summary>The English cardinal for <paramref name="n"/> (negatives and up to &lt; 10^12). Groups
    /// are space-separated, tens–ones hyphenated ("twenty-one"), American style (no "and").</summary>
    internal static string SpellCardinal(long n)
    {
        if (n == 0)
        {
            return "zero";
        }

        if (n < 0)
        {
            return "minus " + SpellCardinal(-n);
        }

        if (n >= 1_000_000_000_000L)
        {
            return n.ToString(CultureInfo.InvariantCulture);
        }

        var parts = new List<string>();
        foreach (var (value, name) in Scales)
        {
            if (n >= value)
            {
                parts.Add(BelowThousand((int)(n / value)) + " " + name);
                n %= value;
            }
        }

        if (n > 0)
        {
            parts.Add(BelowThousand((int)n));
        }

        return string.Join(" ", parts);
    }

    private static readonly (long Value, string Name)[] Scales =
        [(1_000_000_000, "billion"), (1_000_000, "million"), (1000, "thousand")];

    private static string BelowThousand(int n)
    {
        var hundreds = n / 100;
        var rest = n % 100;
        if (hundreds == 0)
        {
            return BelowHundred(rest);
        }

        var word = Ones[hundreds] + " hundred";
        return rest == 0 ? word : word + " " + BelowHundred(rest);
    }

    private static string BelowHundred(int n)
    {
        if (n < 20)
        {
            return Ones[n];
        }

        var unit = n % 10;
        return unit == 0 ? Tens[n / 10] : Tens[n / 10] + "-" + Ones[unit];
    }

    /// <summary>The English ordinal word for <paramref name="n"/>. Only the last group is made
    /// ordinal ("one hundred twenty-first"), and a round scale takes the "-th" on its scale word
    /// ("one hundredth", "one thousandth").</summary>
    internal static string SpellOrdinal(int n)
    {
        if (n < 20)
        {
            return OrdinalOnes[n];
        }

        if (n < 100)
        {
            var unit = n % 10;
            return unit == 0 ? OrdinalTens[n / 10] : Tens[n / 10] + "-" + OrdinalOnes[unit];
        }

        var rest = n % 100;
        return rest == 0
            ? SpellCardinal(n) + "th"                         // one hundredth, one thousandth
            : SpellCardinal(n - rest) + " " + SpellOrdinal(rest);   // one hundred twenty-first
    }

    [GeneratedRegex(@"(?<sym>[$£€])\s*(?<amt>\d{1,3}(?:,\d{3})*|\d+)(?:\.(?<cents>\d{2}))?")]
    private static partial Regex CurrencyRegex();

    [GeneratedRegex(@"(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*%")]
    private static partial Regex PercentRegex();

    [GeneratedRegex(@"(-?\d+(?:\.\d+)?)\s*°\s?C\b")]
    private static partial Regex TemperatureRegex();

    [GeneratedRegex(@"(-?\d+(?:\.\d+)?)\s*°(?!\s?C)")]
    private static partial Regex DegreeRegex();

    [GeneratedRegex(@"\b(?<h>\d{1,2}):(?<m>\d{2})\b(?:\s*(?<ap>[ap]\.?m\.?))?", RegexOptions.IgnoreCase)]
    private static partial Regex TimeRegex();

    [GeneratedRegex(@"\b(\d+)(?:st|nd|rd|th)\b")]
    private static partial Regex OrdinalRegex();

    [GeneratedRegex(@"(?<pre>^|\s|\()-(?=\d)")]
    private static partial Regex MinusRegex();

    [GeneratedRegex(@"\d{1,3}(?:,\d{3})+")]
    private static partial Regex ThousandsRegex();

    [GeneratedRegex(@"(\d+)\.(\d+)")]
    private static partial Regex DecimalRegex();

    [GeneratedRegex(@"\d+")]
    private static partial Regex IntegerRegex();
}
