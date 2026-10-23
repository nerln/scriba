import Foundation

/// Every word of a transcript with its time and its speaker, as the engine writes
/// it to words.json: `[[start, end, "word", "SPEAKER_00"], ...]`.
///
/// This is what lets the player light up the word being heard and jump to a word
/// that is clicked. The documents keep whole turns, which are too coarse for either.
struct WordTrack: Equatable {
    struct Word: Equatable {
        let start: Double
        let end: Double
        let text: String
        let speaker: String?
    }

    /// A run of words by one speaker, the way the text is laid out on screen.
    struct Paragraph: Equatable {
        let speaker: String?
        let start: Double
        /// Indices into `words`.
        let range: Range<Int>
    }

    let words: [Word]

    init(words: [Word]) {
        self.words = words
    }

    /// Reads words.json. Entries that are not `[number, number, string, string?]`
    /// are skipped rather than failing the whole file: one odd word must not take
    /// the player away.
    init(json data: Data) throws {
        let raw = try JSONSerialization.jsonObject(with: data)
        guard let rows = raw as? [[Any]] else {
            throw CocoaError(.fileReadCorruptFile)
        }
        words = rows.compactMap { row in
            guard row.count >= 3,
                  let start = (row[0] as? NSNumber)?.doubleValue,
                  let end = (row[1] as? NSNumber)?.doubleValue,
                  let text = row[2] as? String else { return nil }
            let speaker = row.count > 3 ? row[3] as? String : nil
            return Word(start: start, end: end, text: text, speaker: speaker)
        }
    }

    /// The words grouped into paragraphs: a new one when the speaker changes, or
    /// when the same speaker pauses longer than `pause` seconds, so a monologue
    /// does not become one block of text with no place for the eye to rest.
    func paragraphs(pause: Double = 4.0) -> [Paragraph] {
        var out: [Paragraph] = []
        var first = 0
        for i in words.indices where i > 0 {
            let prev = words[i - 1], word = words[i]
            if word.speaker != prev.speaker || word.start - prev.end > pause {
                out.append(Paragraph(speaker: words[first].speaker,
                                     start: words[first].start, range: first..<i))
                first = i
            }
        }
        if !words.isEmpty {
            out.append(Paragraph(speaker: words[first].speaker,
                                 start: words[first].start, range: first..<words.count))
        }
        return out
    }

    /// The word being heard at `time`: the last one that has started. Between two
    /// words it stays on the one before, so the highlight does not flicker off in
    /// every pause.
    func index(at time: Double) -> Int? {
        guard let firstWord = words.first, time >= firstWord.start else { return nil }
        var lo = 0, hi = words.count - 1
        while lo < hi {
            let mid = (lo + hi + 1) / 2
            if words[mid].start <= time { lo = mid } else { hi = mid - 1 }
        }
        return lo
    }

    var duration: Double { words.last?.end ?? 0 }
}

/// "1:02:05" over an hour, "02:05" under.
func playerClock(_ seconds: Double) -> String {
    let s = max(0, Int(seconds.rounded(.down)))
    if s >= 3600 {
        return String(format: "%d:%02d:%02d", s / 3600, (s % 3600) / 60, s % 60)
    }
    return String(format: "%02d:%02d", s / 60, s % 60)
}
