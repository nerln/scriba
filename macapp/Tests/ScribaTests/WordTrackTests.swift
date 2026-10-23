import Foundation
import Testing

@testable import Scriba

/// The read-along player's arithmetic: which word is heard, and how the text is laid out.
struct WordTrackTests {
    static let json = """
    [[0.0, 0.4, "Hi", "SPEAKER_00"], [0.5, 0.9, "Ada.", "SPEAKER_00"],
     [1.2, 1.5, "Hello", "SPEAKER_01"], [1.6, 2.0, "Rafiq", "SPEAKER_01"],
     [9.0, 9.4, "So", "SPEAKER_01"], ["bad"], [10.0, 10.5, "anyway", null]]
    """

    static var track: WordTrack { try! WordTrack(json: Data(json.utf8)) }

    @Test("words.json is read, and a malformed entry is skipped rather than fatal")
    func reads() {
        #expect(Self.track.words.count == 6)
        #expect(Self.track.words[2] == .init(start: 1.2, end: 1.5, text: "Hello", speaker: "SPEAKER_01"))
        #expect(Self.track.words.last?.speaker == nil)
        #expect(Self.track.duration == 10.5)
    }

    @Test("a paragraph per speaker turn, and a long pause starts a new one")
    func paragraphs() {
        let p = Self.track.paragraphs(pause: 4.0)
        #expect(p.map(\.range) == [0..<2, 2..<4, 4..<5, 5..<6])
        #expect(p[1].speaker == "SPEAKER_01" && p[1].start == 1.2)
    }

    @Test("the lit word is the last one started, and stays lit through a pause")
    func current() {
        let t = Self.track
        #expect(t.index(at: -1) == nil)
        #expect(t.index(at: 0.0) == 0)
        #expect(t.index(at: 0.45) == 0)
        #expect(t.index(at: 1.3) == 2)
        #expect(t.index(at: 5.0) == 3)
        #expect(t.index(at: 99) == 5)
    }

    @Test("every word's characters are where the layout says they are")
    func layout() {
        let t = Self.track
        let (text, ranges) = ReadAlongText.layout(t, names: ["SPEAKER_00": "Ada"])
        let string = text.string as NSString
        for (i, word) in t.words.enumerated() {
            #expect(string.substring(with: ranges[i]) == word.text)
        }
        #expect(text.string.hasPrefix("Ada  00:00\nHi Ada."))
        #expect(text.string.contains("Voice 2  00:01\nHello Rafiq"))
    }

    @Test("the clock shows hours only when there are some")
    func clock() {
        #expect(playerClock(65) == "01:05")
        #expect(playerClock(3725) == "1:02:05")
    }
}
