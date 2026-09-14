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
            parts.append(failed.count == 1 ? "1 did not finish" : "\(failed.count) did not finish")
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
        let minutes = Int(audio.rounded())
        let work = max(1, Int((audio * Self.workShare).rounded()))
        return "\(minutes) min of audio, up to about \(work) min of work."
    }

    /// What the progress panel says under the running recording.
    static func remainingText(_ remaining: Int, continues: Bool) -> String? {
        guard remaining > 0 else { return nil }
        let more = remaining == 1 ? "1 more" : "\(remaining) more"
        return continues
            ? "\(more) after this one"
            : "\(more) waiting. They do not start on their own: choose Transcribe all (⌘T), "
              + "or open one and put it next."
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
            if let why = Self.problem(with: entry.url) { item.state = .failed(why) }
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

    /// Everything that was running goes back to waiting: a Stop is not a failure.
    mutating func stopAll() {
        for i in items.indices where items[i].state == .running {
            items[i].state = .waiting
        }
    }

    // MARK: - files that are not really there

    /// Why this file cannot be transcribed, before the engine is even asked, or
    /// nil if it can. The same three checks the engine makes (audio.py,
    /// `placeholder`): a cloud placeholder, a size with nothing behind it, and a
    /// container too small to hold a second of audio. Made here so the row can
    /// say so the moment it is added, instead of after a run that fails on it.
    static func problem(with url: URL) -> String? {
        var st = stat()
        guard stat(url.path, &st) == 0 else { return nil }
        let datalessFlag: UInt32 = 0x4000_0000  // SF_DATALESS
        if st.st_flags & datalessFlag != 0 {
            return "A placeholder: the contents are held by a cloud service and are not on this Mac."
        }
        if st.st_size > 0 && st.st_blocks == 0 {
            return "Reports a size and holds no data: a cloud service has its contents."
        }
        if st.st_size > 0 && st.st_size < 64 * 1024 {
            return "\(st.st_size / 1024) KB, a header with no audio behind it. "
                 + "The copy was made before the recording had been downloaded."
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
