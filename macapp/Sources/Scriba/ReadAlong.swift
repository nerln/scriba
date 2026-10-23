import SwiftUI
import AVFoundation
import AppKit

/// Play a recording with its transcript, the word being heard lit up, and any word
/// a click away from being heard.
///
/// Like the transcripts in Voice Memos, with the speakers kept: a transcript read
/// on its own is where a misheard name or a figure goes unchecked, and finding the
/// moment in a waveform by ear takes longer than the reading did.
struct ReadAlongView: View {
    let jobDir: String
    let audio: String
    let source: String
    let names: [String: String]

    @StateObject private var player = ReadAlongPlayer()
    @State private var track: WordTrack?
    @State private var problem: String?
    @State private var follow = true
    @State private var scrubbing: Double?
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            controls
                .padding(.horizontal, 18)
                .padding(.vertical, 12)
            Divider()
            if let track {
                ReadAlongText(track: track, names: names,
                              current: player.isLoaded ? track.index(at: scrubbing ?? player.time) : nil,
                              follow: follow,
                              onWord: { i in player.seek(to: track.words[i].start, play: true) })
            } else if let problem {
                Label(problem, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                    .padding(24)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ProgressView("Reading the transcript…")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .task { await load() }
        .onDisappear { player.stop() }
    }

    private var controls: some View {
        HStack(spacing: 12) {
            Button {
                player.toggle()
            } label: {
                Image(systemName: player.isPlaying ? "pause.fill" : "play.fill")
                    .frame(width: 18)
            }
            .keyboardShortcut(.space, modifiers: [])
            .disabled(!player.isLoaded)
            .help(player.isPlaying ? "Pause (space)" : "Play (space)")

            Button { player.skip(-10) } label: { Image(systemName: "gobackward.10") }
                .disabled(!player.isLoaded)
            Button { player.skip(10) } label: { Image(systemName: "goforward.10") }
                .disabled(!player.isLoaded)

            Text(playerClock(scrubbing ?? player.time))
                .font(.system(.callout, design: .monospaced))
                .monospacedDigit()
            Slider(value: Binding(get: { scrubbing ?? player.time },
                                  set: { scrubbing = $0 }),
                   in: 0...max(player.duration, 1)) { editing in
                // Seek once, where the thumb is let go: seeking on every step of a
                // drag stutters the audio and floods the text with scrolls.
                if !editing, let t = scrubbing {
                    player.seek(to: t, play: player.isPlaying)
                    scrubbing = nil
                }
            }
            .disabled(!player.isLoaded)
            Text(playerClock(player.duration))
                .font(.system(.callout, design: .monospaced))
                .foregroundStyle(.secondary)
                .monospacedDigit()

            Toggle("Follow", isOn: $follow)
                .toggleStyle(.checkbox)
                .help("Keep the word being heard in view")
            Button("Done") { dismiss() }
                .keyboardShortcut(.cancelAction)
        }
    }

    private func load() async {
        // The audio the engine converted is on the transcript's clock. When it was
        // pruned to save space, the recording itself has the same timeline.
        let audioURL = FileManager.default.fileExists(atPath: audio)
            ? URL(fileURLWithPath: audio) : URL(fileURLWithPath: source)
        player.load(audioURL)
        if !player.isLoaded {
            problem = "Neither the converted audio nor the recording could be opened."
        }
        guard let url = await WordTrackFile.locate(jobDir: jobDir, source: source),
              let data = try? Data(contentsOf: url),
              let loaded = try? WordTrack(json: data), !loaded.words.isEmpty else {
            problem = "This recording has no word timings to follow. Run it again with "
                    + "alignment on, or read it in the document."
            return
        }
        track = loaded
    }
}

/// words.json in the job folder, written by the engine for runs from now on, and
/// for older ones by `scriba words`, which reads the cache and loads no model.
enum WordTrackFile {
    @MainActor
    static func locate(jobDir: String, source: String) async -> URL? {
        let url = URL(fileURLWithPath: jobDir).appendingPathComponent("words.json")
        if FileManager.default.fileExists(atPath: url.path) { return url }
        let python = Engine.pythonPath
        let root = Engine.enginePath
        return await Task.detached(priority: .userInitiated) { () -> URL? in
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: python)
            proc.arguments = ["-m", "scriba.cli", "words", source]
            proc.currentDirectoryURL = URL(fileURLWithPath: root)
            var env = ProcessInfo.processInfo.environment
            env["PYTHONPATH"] = root
            proc.environment = env
            proc.standardOutput = FileHandle.nullDevice
            proc.standardError = FileHandle.nullDevice
            guard (try? proc.run()) != nil else { return nil }
            proc.waitUntilExit()
            return FileManager.default.fileExists(atPath: url.path) ? url : nil
        }.value
    }
}

/// The audio side: a player and a clock the text can follow.
@MainActor
final class ReadAlongPlayer: ObservableObject {
    @Published private(set) var time: Double = 0
    @Published private(set) var isPlaying = false
    @Published private(set) var duration: Double = 0
    @Published private(set) var isLoaded = false

    private var player: AVAudioPlayer?
    private var ticker: Timer?

    func load(_ url: URL) {
        guard let p = try? AVAudioPlayer(contentsOf: url) else { return }
        p.prepareToPlay()
        player = p
        duration = p.duration
        isLoaded = true
    }

    func toggle() {
        guard let player else { return }
        if player.isPlaying { pause() } else { play() }
    }

    func play() {
        guard let player else { return }
        player.play()
        isPlaying = true
        // A tenth of a second: a word lasts about a third of one, so the highlight
        // never lags a word behind, and the text is not redrawn more than it needs.
        ticker?.invalidate()
        ticker = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.tick() }
        }
    }

    func pause() {
        player?.pause()
        isPlaying = false
        ticker?.invalidate()
        ticker = nil
        tick()
    }

    func stop() {
        pause()
        player = nil
    }

    func seek(to seconds: Double, play: Bool) {
        guard let player else { return }
        player.currentTime = min(max(0, seconds), player.duration)
        time = player.currentTime
        if play, !player.isPlaying { self.play() }
    }

    func skip(_ seconds: Double) {
        seek(to: time + seconds, play: isPlaying)
    }

    private func tick() {
        guard let player else { return }
        time = player.currentTime
        if !player.isPlaying, isPlaying {
            // Reached the end.
            isPlaying = false
            ticker?.invalidate()
            ticker = nil
        }
    }
}

/// The transcript as one text view: tens of thousands of words in an hour, which
/// SwiftUI cannot lay out as separate buttons without stalling the window. A click
/// that selects nothing plays from the word under it; a drag still selects text to
/// copy.
struct ReadAlongText: NSViewRepresentable {
    let track: WordTrack
    let names: [String: String]
    let current: Int?
    let follow: Bool
    let onWord: (Int) -> Void

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSTextView.scrollableTextView()
        guard let text = scroll.documentView as? NSTextView else { return scroll }
        let view = WordTextView(frame: text.frame, textContainer: text.textContainer)
        view.isEditable = false
        view.isSelectable = true
        view.drawsBackground = false
        view.textContainerInset = NSSize(width: 24, height: 18)
        view.autoresizingMask = text.autoresizingMask
        view.minSize = text.minSize
        view.maxSize = text.maxSize
        view.isVerticallyResizable = true
        scroll.documentView = view
        scroll.drawsBackground = false

        let (string, ranges) = Self.layout(track, names: names)
        view.textStorage?.setAttributedString(string)
        context.coordinator.ranges = ranges
        view.onClick = { [weak coordinator = context.coordinator] index in
            guard let coordinator, let word = coordinator.word(at: index) else { return }
            onWord(word)
        }
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        guard let view = scroll.documentView as? WordTextView,
              let storage = view.textStorage else { return }
        let c = context.coordinator
        view.onClick = { [weak c] index in
            guard let c, let word = c.word(at: index) else { return }
            onWord(word)
        }
        guard current != c.lit else { return }
        storage.beginEditing()
        if let old = c.lit, old < c.ranges.count {
            storage.removeAttribute(.backgroundColor, range: c.ranges[old])
        }
        if let new = current, new < c.ranges.count {
            storage.addAttribute(.backgroundColor,
                                 value: NSColor.controlAccentColor.withAlphaComponent(0.3),
                                 range: c.ranges[new])
        }
        storage.endEditing()
        c.lit = current
        if follow, let new = current, new < c.ranges.count {
            view.scrollRangeToVisible(c.ranges[new])
        }
    }

    final class Coordinator {
        var ranges: [NSRange] = []
        var lit: Int?

        /// The word whose characters contain `index`, or the nearest one before it.
        func word(at index: Int) -> Int? {
            guard !ranges.isEmpty else { return nil }
            var lo = 0, hi = ranges.count - 1
            while lo < hi {
                let mid = (lo + hi + 1) / 2
                if ranges[mid].location <= index { lo = mid } else { hi = mid - 1 }
            }
            return lo
        }
    }

    /// The text, one paragraph per speaker turn under the speaker's name and time,
    /// and the character range of every word in it.
    static func layout(_ track: WordTrack, names: [String: String]) -> (NSAttributedString, [NSRange]) {
        let out = NSMutableAttributedString()
        var ranges = [NSRange](repeating: NSRange(location: 0, length: 0), count: track.words.count)
        let body: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 15), .foregroundColor: NSColor.labelColor]
        let label: [NSAttributedString.Key: Any] = [
            .font: NSFont.boldSystemFont(ofSize: 12), .foregroundColor: NSColor.secondaryLabelColor]
        for (n, paragraph) in track.paragraphs().enumerated() {
            if n > 0 { out.append(NSAttributedString(string: "\n\n", attributes: body)) }
            let who = paragraph.speaker.map { names[$0] ?? Speaker.label(for: $0) } ?? "?"
            out.append(NSAttributedString(string: "\(who)  \(playerClock(paragraph.start))\n",
                                          attributes: label))
            for i in paragraph.range {
                if i > paragraph.range.lowerBound {
                    out.append(NSAttributedString(string: " ", attributes: body))
                }
                let text = track.words[i].text
                ranges[i] = NSRange(location: out.length, length: (text as NSString).length)
                out.append(NSAttributedString(string: text, attributes: body))
            }
        }
        return (out, ranges)
    }
}

/// A text view that reports a plain click, and leaves drags to selection.
final class WordTextView: NSTextView {
    var onClick: ((Int) -> Void)?

    override func mouseDown(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        let index = characterIndexForInsertion(at: point)
        // NSTextView tracks the drag itself and returns at mouse up. Nothing
        // selected by then means it was a click.
        super.mouseDown(with: event)
        if selectedRange().length == 0, event.clickCount == 1 {
            onClick?(index)
        }
    }
}
