import Foundation

/// The queue as a value: what is in it, what to call it, what to save.
///
/// The window owns one of these. Pulling the arithmetic out of the view is what
/// lets it be tested, and it is arithmetic somebody got wrong once already: the
/// section header counted the running row and the failed rows as waiting.
struct Queue: Equatable {
    var items: [QueueItem] = []

    /// Minutes of work per minute of recording, on this Mac, with the engine on
    /// Metal. Measured on 14 September 2026 on two meetings: 26 minutes of audio
    /// took 12.5 minutes end to end, 79 minutes took 19. Half is the cautious
    /// end of that, and a number that comes in early is a number nobody minds.
    static let workShare = 0.5

    // MARK: - what is in it

    var isEmpty: Bool { items.isEmpty }
    var waiting: [QueueItem] { items.filter { $0.state == .waiting } }
    var running: QueueItem? { items.first { $0.state == .running } }
    var failed: [QueueItem] { items.filter { $0.state.isFailed } }
    var nextToRun: QueueItem? { waiting.first }

    /// The rows in the order the sidebar draws them: the running one, then
    /// what waits in the order it will run, then the ones with a problem.
    ///
    /// The list used to follow the order things were added. A recording
    /// started from the bottom of a long queue ran down there, under eight
    /// rows of files that were never on this Mac, and the one row that was
    /// moving could not be seen without scrolling.
    var shown: [QueueItem] {
        items.filter { $0.state == .running } + waiting
            + items.filter { $0.state == .finished } + failed
    }

    /// The files in it, spelled the way the engine spells them.
    var paths: Set<String> { Set(items.map(\.path)) }

    func index(of id: UUID) -> Int? { items.firstIndex { $0.id == id } }
    func item(_ id: UUID) -> QueueItem? { items.first { $0.id == id } }

    // MARK: - what to call it

    /// The section header. It counts what is waiting, and says separately what
    /// is running and what stopped, because those are three different answers
    /// to "is it working".
    var title: String {
        var parts: [String] = []
        if running != nil {
            parts.append(waiting.isEmpty ? "Transcribing" : "Transcribing, \(waiting.count) waiting")
        } else if !waiting.isEmpty {
            parts.append("Waiting to be transcribed (\(waiting.count))")
        }
        if !failed.isEmpty {
            // "With a problem" rather than "did not finish": some of these never
            // started, because the file had moved or was not audio.
            parts.append(failed.count == 1 ? "1 with a problem" : "\(failed.count) with a problem")
        }
        if parts.isEmpty { return "Transcribed" }
        return parts.joined(separator: " · ")
    }

    /// The label of the button that starts everything, or nil when there is
    /// nothing to start.
    var startTitle: String? {
        switch waiting.count {
        case 0: return nil
        case 1: return "Transcribe this one"
        default: return "Transcribe all \(waiting.count)"
        }
    }

    /// How long the waiting recordings will take, said before they start.
    /// Somebody who does not know that a transcription runs for minutes reads a
    /// still progress bar as a hang.
    var estimate: String {
        let waiting = self.waiting
        guard !waiting.isEmpty else { return "" }
        let audio = waiting.compactMap(\.minutes).reduce(0, +)
        guard audio > 0 else { return "Takes up to half the length of the recordings." }
        let work = max(1, Int((audio * Self.workShare).rounded()))
        // Under a minute in seconds. Rounded to minutes, a batch of short
        // memos read "0 min of audio, up to about 1 min of work".
        let length = audio < 0.5 ? "\(max(1, Int((audio * 60).rounded()))) s"
                                 : "\(Int(audio.rounded())) min"
        return "\(length) of audio, up to about \(work) min of work."
    }

    /// What the progress panel says under the running recording. `asked` is the
    /// recordings somebody put next: they run, in that order, and the queue stops
    /// again behind them.
    static func remainingText(_ remaining: Int, continues: Bool,
                              asked: [String] = []) -> String? {
        guard remaining > 0 else { return nil }
        let more = remaining == 1 ? "1 more" : "\(remaining) more"
        if continues { return "\(more) after this one" }
        let rest = "choose Transcribe all (⌘T), or open one and put it next."
        guard !asked.isEmpty else {
            return "\(more) waiting. They do not start on their own: " + rest
        }
        let goes = asked.count == 1 ? "\(asked[0]) goes next"
                                    : "\(asked.count) go next, in the order they were asked for"
        let others = remaining - asked.count
        guard others > 0 else { return goes + ", and then it stops." }
        let waits = others == 1 ? "The other one does not start on its own: "
                                : "The other \(others) do not start on their own: "
        return goes + ", and then it stops. " + waits + rest
    }

    /// The one message a batch gives when it ends, or nil if nothing failed.
    /// The reasons are on the rows as well; this is the place they are read
    /// together, once, instead of one alert per file.
    static func batchSummary(_ failures: [(name: String, reason: String)]) -> String? {
        guard !failures.isEmpty else { return nil }
        let head = failures.count == 1 ? "1 recording did not go through:"
                                       : "\(failures.count) recordings did not go through:"
        return ([head] + failures.map { "\($0.name): \($0.reason)" }).joined(separator: "\n")
    }

    /// Whether the Transcribe button on a queued recording does anything.
    static func canStart(_ state: QueueItem.State, engineBusy: Bool) -> Bool {
        switch state {
        case .waiting, .failed: return !engineBusy
        case .running, .finished: return false
        }
    }

    // MARK: - changing it

    /// Add the ones not already listed, in the order given. Returns what was
    /// added, so the caller can go and measure them.
    @discardableResult
    mutating func add(_ found: [(url: URL, collection: String)],
                      language: String = "auto", speakers: Int = 0) -> [QueueItem] {
        let known = paths
        var added: [QueueItem] = []
        for entry in found where !known.contains(canonicalPath(entry.url.path)) {
            var item = QueueItem(url: entry.url, collection: entry.collection)
            item.language = language
            item.speakers = speakers
            item.note = Self.note(for: entry.url)
            added.append(item)
        }
        items.append(contentsOf: added)
        return added
    }

    mutating func mark(_ id: UUID, _ state: QueueItem.State) {
        guard let i = index(of: id) else { return }
        items[i].state = state
    }

    /// Take a row out. Never the running one: the engine is on it.
    mutating func remove(_ id: UUID) {
        items.removeAll { $0.id == id && $0.state != .running }
    }

    /// Put a waiting recording first in line.
    mutating func moveToFront(_ id: UUID) {
        guard let i = index(of: id), items[i].state == .waiting else { return }
        let item = items.remove(at: i)
        let firstWaiting = items.firstIndex { $0.state == .waiting } ?? items.endIndex
        items.insert(item, at: firstWaiting)
    }

    /// Drop the inbox rows the engine has got to by another road.
    ///
    /// A recording from the inbox can be transcribed from a terminal, or by
    /// `scriba memos`, while it sits here as waiting. Once a job for it exists
    /// the row is stale: the job list shows the run, or the document, and a row
    /// that keeps saying "waiting" over a finished transcript is the kind of
    /// thing that makes a queue look broken. Rows added by hand are left alone:
    /// dropping a file that already has a job is how you ask for it again.
    /// Returns the ids taken out, so the caller can fix the selection.
    @discardableResult
    mutating func reconcileInbox(known: Set<String>) -> [UUID] {
        let stale = items.filter {
            $0.collection == Inbox.collection && $0.state == .waiting && known.contains($0.path)
        }
        let ids = Set(stale.map(\.id))
        items.removeAll { ids.contains($0.id) }
        return stale.map(\.id)
    }

    /// Everything that was running goes back to waiting: a Stop is not a failure.
    mutating func stopAll() {
        for i in items.indices where items[i].state == .running {
            items[i].state = .waiting
        }
    }

    // MARK: - files that may not be what they seem

    /// What to say about this file before the engine is asked, or nil.
    ///
    /// It says, and does not refuse. The rows used to be marked failed here, with
    /// a reason that was a guess: a ten-second voice note at 32 kbps is forty
    /// kilobytes and was called "a header with no audio behind it", and a file
    /// that iCloud had moved off this Mac was called "not on this Mac" although
    /// reading it, which is the first thing the engine does, makes the system
    /// fetch it. A failed row is skipped by Transcribe all, so both were never
    /// transcribed. The engine opens the file first and explains itself only if
    /// that fails (audio.py, `placeholder`); this leaves the decision there.
    static func note(for url: URL) -> QueueItem.Note? {
        var st = stat()
        guard stat(url.path, &st) == 0 else { return nil }
        let datalessFlag: UInt32 = 0x4000_0000  // SF_DATALESS
        if st.st_flags & datalessFlag != 0 || (st.st_size > 0 && st.st_blocks == 0) {
            return QueueItem.Note(text: "The contents are held by a cloud service and are not on "
                + "this Mac yet. Starting it makes the system fetch them first, which can "
                + "take a while. If they cannot be fetched the run says so.")
        }
        if st.st_size > 0 && st.st_size < 64 * 1024 {
            return QueueItem.Note(text: "\(max(1, st.st_size / 1024)) KB is small for a "
                + "recording. A short voice note can be this size; a copy made before the "
                + "download had finished looks the same, and if so the run says so.")
        }
        return nil
    }

    // MARK: - surviving a quit

    /// What is worth writing down: the waiting rows and the running one, which
    /// comes back as waiting because quitting killed its run. Finished rows are
    /// about to leave and failed rows carry a reason that will not be true
    /// tomorrow.
    struct Saved: Codable, Equatable {
        let path: String
        var collection: String = ""
        var language: String = "auto"
        var speakers: Int = 0
    }

    var saved: [Saved] {
        items.filter { $0.state == .waiting || $0.state == .running }
            .map { Saved(path: $0.url.path, collection: $0.collection,
                         language: $0.language, speakers: $0.speakers) }
    }

    /// The rows to put back, skipping files that are no longer there.
    static func restore(_ saved: [Saved],
                        exists: (String) -> Bool = { FileManager.default.fileExists(atPath: $0) }) -> Queue {
        var queue = Queue()
        for entry in saved where exists(entry.path) {
            let url = URL(fileURLWithPath: entry.path)
            queue.add([(url: url, collection: entry.collection)],
                      language: entry.language, speakers: entry.speakers)
        }
        return queue
    }
}
