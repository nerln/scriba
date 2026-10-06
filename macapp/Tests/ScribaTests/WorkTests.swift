import AVFoundation
import Combine
import Foundation
import Testing

@testable import Scriba

// MARK: - stand-ins

/// Counts what the model asks the power management for, without asking it.
@MainActor
final class FakeAssertion {
    private(set) var begun: [String] = []
    private(set) var ended = 0

    /// How many holds are on right now.
    var held: Int { begun.count - ended }

    func awake() -> StayAwake {
        StayAwake(begin: { reason in
            self.begun.append(reason)
            return NSObject()
        }, end: { _ in
            self.ended += 1
        })
    }
}

/// Says what it was asked and shows nothing.
@MainActor
final class FakeNotifier: BatchNotifying {
    private(set) var asked = 0
    private(set) var posted: [BatchNotice.Text] = []

    func askIfUndecided() { asked += 1 }
    func post(_ text: BatchNotice.Text) { posted.append(text) }
}

// MARK: - staying awake

@MainActor
struct StayAwakeTests {

    @Test("asking twice holds once, and letting go twice lets go once")
    func holdsOnce() {
        let power = FakeAssertion()
        let awake = power.awake()
        awake.set(true)
        awake.set(true)
        #expect(power.begun.count == 1)
        #expect(awake.isHeld)
        awake.set(false)
        awake.set(false)
        #expect(power.ended == 1)
        #expect(!awake.isHeld)
        #expect(power.held == 0)
    }

    @Test("letting go with nothing held asks nothing of the system")
    func releaseWithoutHold() {
        let power = FakeAssertion()
        power.awake().release()
        #expect(power.ended == 0)
    }

    @Test("it can be held again after it was let go")
    func holdsAgain() {
        let power = FakeAssertion()
        let awake = power.awake()
        awake.hold(); awake.release(); awake.hold()
        #expect(power.begun.count == 2)
        #expect(power.held == 1)
    }

    @Test("the reason the system shows is one fixed sentence about no recording in particular")
    func fixedReason() {
        let power = FakeAssertion()
        let awake = power.awake()
        awake.hold()
        #expect(power.begun == ["Scriba is recording or transcribing"])
    }

    @Test("the real assertion is on the system while held, and gone when let go")
    func realAssertion() async throws {
        let awake = StayAwake()
        defer { awake.release() }
        awake.hold()
        #expect(await eventually(10) { Self.systemShows(StayAwake.reason) })
        awake.release()
        #expect(await eventually(10) { !Self.systemShows(StayAwake.reason) })
    }

    /// Whether `pmset` lists an assertion with this name. Reading it changes nothing.
    nonisolated static func systemShows(_ reason: String) -> Bool {
        let pmset = Process()
        pmset.executableURL = URL(fileURLWithPath: "/usr/bin/pmset")
        pmset.arguments = ["-g", "assertions"]
        let out = Pipe()
        pmset.standardOutput = out
        pmset.standardError = FileHandle.nullDevice
        guard (try? pmset.run()) != nil else { return false }
        let data = out.fileHandleForReading.readDataToEndOfFile()
        pmset.waitUntilExit()
        return String(decoding: data, as: UTF8.self).contains("named: \"\(reason)\"")
    }
}

// MARK: - the notification

struct BatchNoticeTests {

    @Test("a batch with one failure says both numbers")
    func bothNumbers() {
        let text = BatchNotice.text(for: [.finished, .finished, .failed, .finished])
        #expect(text?.body == "3 transcribed, 1 did not go through")
    }

    @Test("a clean batch says only what went through")
    func allWent() {
        #expect(BatchNotice.text(for: [.finished, .finished])?.body == "2 transcribed")
        #expect(BatchNotice.text(for: [.finished])?.body == "1 transcribed")
    }

    @Test("a batch where nothing went through says only that")
    func noneWent() {
        #expect(BatchNotice.text(for: [.failed, .failed])?.body == "2 did not go through")
    }

    @Test("nothing to say gives no notification")
    func nothing() {
        #expect(BatchNotice.text(for: []) == nil)
    }

    @Test("a recording stopped by hand is not counted")
    func stoppedIsNotCounted() {
        #expect(BatchNotice.text(for: [.cancelled]) == nil)
        #expect(BatchNotice.text(for: [.finished, .cancelled])?.body == "1 transcribed")
    }

    @Test("it is numbers and fixed words, and the title is the app's name")
    func numbersOnly() throws {
        let text = try #require(BatchNotice.text(for: [.finished, .failed, .failed]))
        #expect(text.title == "Scriba")
        // Whatever the counts, nothing else can be in it: no file name, no reason.
        let words = text.body.replacingOccurrences(of: ",", with: "").split(separator: " ")
        let allowed: Set<String> = ["transcribed", "did", "not", "go", "through"]
        #expect(words.allSatisfy { Int($0) != nil || allowed.contains(String($0)) })
    }
}

/// The real notifier, where there is no application to notify from.
@MainActor
struct SystemNotifierTests {
    @Test("outside an application bundle it asks nothing and shows nothing, and does not end the process")
    func quietWithoutABundle() async {
        // The test runner is not an .app. `UNUserNotificationCenter.current()` from
        // here is an exception that ends the process, so the notifier must not ask.
        let notifier = SystemNotifier()
        notifier.askIfUndecided()
        notifier.post(BatchNotice.Text(title: "Scriba", body: "1 transcribed"))
        await idle(0.1)
    }
}

// MARK: - the input changing under a recording

/// An audio engine that does what the hardware does to a real one: the input can
/// change, the engine stops, the system says so, and nothing flows until it is
/// started again.
final class FakeEngine: RecordingEngine {
    var format: AVAudioFormat?
    var startFailures = 0
    private(set) var starts = 0
    private(set) var stops = 0
    private(set) var tapFormat: AVAudioFormat?
    private var block: AVAudioNodeTapBlock?
    private(set) var running = false
    var onConfigurationChange: (() -> Void)?

    init(format: AVAudioFormat?) { self.format = format }

    var inputFormat: AVAudioFormat? { format }

    func installTap(format: AVAudioFormat, block: @escaping AVAudioNodeTapBlock) {
        tapFormat = format
        self.block = block
    }

    func removeTap() { block = nil }

    func start() throws {
        starts += 1
        if startFailures > 0 {
            startFailures -= 1
            throw NSError(domain: "fake", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "The device is busy."])
        }
        running = true
    }

    func stop() { running = false; stops += 1 }

    /// What plugging something in does: the engine stops, the input is now
    /// `new`, and the system posts its notice.
    func changeInput(to new: AVAudioFormat?, notify: Bool = true) {
        running = false
        format = new
        if notify { onConfigurationChange?() }
    }

    /// The microphone hands over `seconds` of sound, if the engine is running
    /// and a tap is in. Otherwise nothing arrives, as on a stopped engine.
    func deliver(seconds: Double, amplitude: Float = 0.3) {
        guard running, let block, let format else { return }
        block(RecorderTests.tone(format, seconds: seconds, amplitude: amplitude),
              AVAudioTime(sampleTime: 0, atRate: format.sampleRate))
    }
}

/// A live transcriber that records what it was given.
final class FakeLive: LiveTranscribing {
    var onFinal: ((String) -> Void)?
    var onGuess: ((String) -> Void)?
    var canFollow = true
    /// How long finishing takes, as the real analyzer takes about a second.
    var finishTakes: Double = 0
    private(set) var fed = 0
    private(set) var followed: [Double] = []
    private(set) var finished = false

    func feed(_ buffer: AVAudioPCMBuffer) { fed += 1 }
    func follow(_ format: AVAudioFormat) -> Bool {
        followed.append(format.sampleRate)
        return canFollow
    }
    func finish() async {
        if finishTakes > 0 { try? await Task.sleep(nanoseconds: UInt64(finishTakes * 1_000_000_000)) }
        finished = true
    }
}

@MainActor
struct InputChangeTests {
    static func format(_ rate: Double, _ channels: AVAudioChannelCount = 1) -> AVAudioFormat {
        RecorderTests.tapFormat(rate: rate, channels: channels)
    }

    /// A folder of its own, so nothing is written under the real home.
    /// And a defaults domain of its own: the live text switch is one key, and
    /// tests that run side by side must not turn it on for each other.
    final class Folder {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("scriba-input-\(UUID().uuidString)")
        private let suite = "scriba-input-\(UUID().uuidString)"
        let defaults: UserDefaults
        init() {
            try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            defaults = UserDefaults(suiteName: suite)!
        }
        deinit {
            try? FileManager.default.removeItem(at: url)
            defaults.removePersistentDomain(forName: suite)
        }
    }

    /// A recorder wired to a fake engine, already recording.
    static func recording(_ folder: Folder, engine: FakeEngine, live: FakeLive? = nil)
        async -> Recorder {
        let recorder = Recorder(directory: folder.url, defaults: folder.defaults,
                                makeEngine: { engine },
                                microphoneAllowed: { true },
                                makeLive: { _, _ in live! })
        recorder.restartPause = 0.01
        if live != nil { recorder.liveTextEnabled = true }
        await recorder.start(language: "en")
        return recorder
    }

    /// Sound from the microphone, and a moment for the level readings the tap
    /// hands to the main actor to arrive, as they do while the real thing runs.
    static func play(_ engine: FakeEngine, _ seconds: Double) async {
        engine.deliver(seconds: seconds)
        await idle(0.05)
    }

    /// The peak of the last `seconds` of a finished file.
    static func loudness(of url: URL, lastSeconds seconds: Double) throws -> Float {
        let file = try AVAudioFile(forReading: url)
        let frames = AVAudioFrameCount(file.processingFormat.sampleRate * seconds)
        file.framePosition = max(file.length - AVAudioFramePosition(frames), 0)
        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: file.processingFormat,
                                                   frameCapacity: frames))
        try file.read(into: buffer, frameCount: frames)
        return Recorder.peak(buffer)
    }

    @Test("a recording on a steady input is written to its file")
    func steadyInput() async throws {
        let folder = Folder()
        let engine = FakeEngine(format: Self.format(48_000))
        let recorder = await Self.recording(folder, engine: engine)
        #expect(recorder.isRecording)
        await Self.play(engine, 1.5)
        let url = try #require(await recorder.stop())
        let file = try AVAudioFile(forReading: url)
        #expect(file.length >= 70_000 && file.length <= 74_000)
        #expect(file.fileFormat.sampleRate == 48_000)
        #expect(folder.url.path == url.deletingLastPathComponent().path)
    }

    @Test("AirPods connecting mid-recording: the tap goes back in on the new format and the same file keeps growing")
    func airPodsConnect() async throws {
        let folder = Folder()
        let engine = FakeEngine(format: Self.format(48_000))
        let recorder = await Self.recording(folder, engine: engine)
        await Self.play(engine, 1.5)

        // The headset's microphone is narrow band, at another rate. The engine
        // stops, and while it is stopped nothing arrives.
        engine.changeInput(to: Self.format(16_000), notify: false)
        await Self.play(engine, 1.0)
        await recorder.inputChanged()

        #expect(engine.running)
        #expect(engine.starts == 2)
        #expect(engine.tapFormat?.sampleRate == 16_000)
        #expect(recorder.isRecording)
        await Self.play(engine, 1.5)
        let url = try #require(await recorder.stop())
        let file = try AVAudioFile(forReading: url)
        // Three seconds from two inputs, in one file, in the file's own format. A
        // little is lost to the resampler holding back its first frames.
        #expect(file.fileFormat.sampleRate == 48_000)
        #expect(file.length >= 3 * 48_000 - 8_000 && file.length <= 3 * 48_000 + 2_000)
        // And the second half is sound, not the silence a stopped engine leaves.
        #expect(try Self.loudness(of: url, lastSeconds: 1.0) > 0.1)
        #expect(recorder.warning == nil)
    }

    @Test("a USB microphone with two channels is mixed down into the same mono file")
    func usbStereo() async throws {
        let folder = Folder()
        let engine = FakeEngine(format: Self.format(48_000))
        let recorder = await Self.recording(folder, engine: engine)
        await Self.play(engine, 1.2)
        engine.changeInput(to: Self.format(44_100, 2), notify: false)
        await recorder.inputChanged()
        await Self.play(engine, 1.2)
        let url = try #require(await recorder.stop())
        let file = try AVAudioFile(forReading: url)
        #expect(file.processingFormat.channelCount == 1)
        #expect(file.length >= Int64(2.4 * 48_000) - 8_000)
        #expect(try Self.loudness(of: url, lastSeconds: 1.0) > 0.1)
    }

    @Test("the system's notice reaches the recorder, which follows by itself")
    func noticeIsObserved() async throws {
        let folder = Folder()
        let engine = FakeEngine(format: Self.format(48_000))
        let recorder = await Self.recording(folder, engine: engine)
        await Self.play(engine, 1.2)
        engine.changeInput(to: Self.format(24_000))      // posts the notice
        #expect(await eventually { engine.starts == 2 })
        #expect(engine.tapFormat?.sampleRate == 24_000)
        await Self.play(engine, 1.2)
        let url = try #require(await recorder.stop())
        #expect(try Self.loudness(of: url, lastSeconds: 1.0) > 0.1)
    }

    @Test("a restart that fails once is tried again, and the recording goes on")
    func restartRetries() async throws {
        let folder = Folder()
        let engine = FakeEngine(format: Self.format(48_000))
        let recorder = await Self.recording(folder, engine: engine)
        await Self.play(engine, 1.2)
        engine.startFailures = 1
        engine.changeInput(to: Self.format(16_000), notify: false)
        await recorder.inputChanged()
        #expect(engine.starts == 3)
        #expect(engine.running)
        #expect(recorder.isRecording)
        #expect(recorder.warning == nil)
        await Self.play(engine, 1.2)
        let url = try #require(await recorder.stop())
        #expect(try Self.loudness(of: url, lastSeconds: 1.0) > 0.1)
    }

    @Test("a restart that never works stops the recording cleanly, keeps the file, and says why")
    func restartFails() async throws {
        let folder = Folder()
        let engine = FakeEngine(format: Self.format(48_000))
        let recorder = await Self.recording(folder, engine: engine)
        await Self.play(engine, 2.0)
        var ended: [URL?] = []
        let watch = recorder.stoppedItself.sink { ended.append($0) }
        defer { watch.cancel() }

        engine.startFailures = 99
        engine.changeInput(to: Self.format(16_000), notify: false)
        await recorder.inputChanged()

        #expect(!recorder.isRecording)
        #expect(recorder.state == .idle)
        #expect(engine.starts == 1 + recorder.restartAttempts)
        let url = try #require(ended.first ?? nil)
        #expect(ended.count == 1)
        // A closed file with the length it was written with, not a file of unknown duration.
        let file = try AVAudioFile(forReading: url)
        #expect(file.length >= 2 * 48_000 - 4_000)
        let warning = try #require(recorder.warning)
        #expect(warning.contains("could not be started again"))
        #expect(warning.contains("The device is busy"))
        #expect(warning.contains("kept"))
    }

    @Test("an input that is gone and not replaced ends the recording with that said")
    func inputGone() async throws {
        let folder = Folder()
        let engine = FakeEngine(format: Self.format(48_000))
        let recorder = await Self.recording(folder, engine: engine)
        await Self.play(engine, 1.5)
        var ended: [URL?] = []
        let watch = recorder.stoppedItself.sink { ended.append($0) }
        defer { watch.cancel() }

        engine.changeInput(to: nil, notify: false)
        await recorder.inputChanged()

        #expect(!recorder.isRecording)
        #expect(try #require(ended.first ?? nil).pathExtension == "wav")
        #expect(recorder.warning?.contains("disconnected") == true)
    }

    @Test("a recording too short to keep is dropped, and the warning says both things")
    func tooShortToKeep() async throws {
        let folder = Folder()
        let engine = FakeEngine(format: Self.format(48_000))
        let recorder = await Self.recording(folder, engine: engine)
        await Self.play(engine, 0.3)
        var ended: [URL?] = []
        let watch = recorder.stoppedItself.sink { ended.append($0) }
        defer { watch.cancel() }

        engine.changeInput(to: nil, notify: false)
        await recorder.inputChanged()

        #expect(ended.map { $0 == nil } == [true])
        #expect(recorder.warning?.contains("disconnected") == true)
        #expect(recorder.warning?.contains("under a second") == true)
        #expect(try FileManager.default.contentsOfDirectory(atPath: folder.url.path).isEmpty)
    }

    @Test("pressing Stop while a restart is waiting to be tried again ends it quietly")
    func stoppedWhileWaiting() async throws {
        let folder = Folder()
        let engine = FakeEngine(format: Self.format(48_000))
        let recorder = await Self.recording(folder, engine: engine)
        recorder.restartPause = 0.3
        await Self.play(engine, 1.5)
        var ended = 0
        let watch = recorder.stoppedItself.sink { _ in ended += 1 }
        defer { watch.cancel() }

        engine.startFailures = 99
        engine.changeInput(to: Self.format(16_000), notify: false)
        let following = Task { await recorder.inputChanged() }
        await idle(0.1)
        let url = await recorder.stop()
        await following.value

        #expect(url != nil)
        #expect(ended == 0)             // somebody stopped it: nobody has to be told it stopped itself
        #expect(!recorder.isRecording)
        #expect(engine.starts <= 2)
    }

    @Test("a second Stop while the first is finishing waits for it, and gets the same file, closed")
    func secondStopWaits() async throws {
        let folder = Folder()
        let engine = FakeEngine(format: Self.format(48_000))
        let live = FakeLive()
        live.finishTakes = 0.4
        let recorder = await Self.recording(folder, engine: engine, live: live)
        await Self.play(engine, 1.5)

        let first = Task { await recorder.stop() }
        await idle(0.1)
        #expect(recorder.isRecording)            // the first is waiting for the live text
        let second = await recorder.stop()
        // Back only now, with the stop over: the file closed at its real length.
        #expect(live.finished)
        #expect(!recorder.isRecording)
        let url = try #require(second)
        #expect(await first.value == url)
        let file = try AVAudioFile(forReading: url)
        #expect(file.length >= 70_000 && file.length <= 74_000)
        // And nothing was deleted as "under a second" on the way.
        #expect(recorder.warning == nil)
        #expect(FileManager.default.fileExists(atPath: url.path))
        // A third, afterwards, has nothing to stop.
        #expect(await recorder.stop() == nil)
    }

    @Test("a stop that is too short to keep tells every caller the same thing")
    func secondStopOnATooShortRecording() async throws {
        let folder = Folder()
        let engine = FakeEngine(format: Self.format(48_000))
        let live = FakeLive()
        live.finishTakes = 0.3
        let recorder = await Self.recording(folder, engine: engine, live: live)
        await Self.play(engine, 0.3)
        let first = Task { await recorder.stop() }
        await idle(0.1)
        #expect(await recorder.stop() == nil)
        #expect(await first.value == nil)
        #expect(recorder.warning?.contains("under a second") == true)
        #expect(try FileManager.default.contentsOfDirectory(atPath: folder.url.path).isEmpty)
    }

    @Test("the live text follows the new input when it can")
    func liveFollows() async throws {
        let folder = Folder()
        let engine = FakeEngine(format: Self.format(48_000))
        let live = FakeLive()
        let recorder = await Self.recording(folder, engine: engine, live: live)
        await Self.play(engine, 1.2)
        let before = live.fed

        engine.changeInput(to: Self.format(16_000), notify: false)
        await recorder.inputChanged()
        await Self.play(engine, 1.2)
        #expect(live.followed == [16_000])
        #expect(live.fed > before)
        #expect(recorder.live.unavailable == nil)
        #expect(!live.finished)
        await recorder.stop()
        #expect(live.finished)
    }

    @Test("when the live text cannot follow, only the live text stops, and the pane says so")
    func liveCannotFollow() async throws {
        let folder = Folder()
        let engine = FakeEngine(format: Self.format(48_000))
        let live = FakeLive()
        let recorder = await Self.recording(folder, engine: engine, live: live)
        await Self.play(engine, 1.2)
        live.canFollow = false
        engine.changeInput(to: Self.format(16_000), notify: false)
        await recorder.inputChanged()
        let before = live.fed
        await Self.play(engine, 1.2)
        #expect(recorder.isRecording)
        #expect(live.fed == before)
        #expect(recorder.live.unavailable?.contains("Live text stopped") == true)
        #expect(await eventually { live.finished })
        let url = try #require(await recorder.stop())
        #expect(try Self.loudness(of: url, lastSeconds: 1.0) > 0.1)
    }

    @Test("a buffer in a format the tap was not built for is dropped, not fed to the converter")
    func mismatchedBufferIsDropped() throws {
        let folder = Folder()
        let fileFormat = RecorderTests.fileFormat()
        let url = try Recorder.reserve(in: folder.url, stamp: "2026-10-06 10-00-00")
        let file = try AVAudioFile(forWriting: url, settings: fileFormat.settings,
                                   commonFormat: .pcmFormatInt16, interleaved: true)
        let input = Self.format(48_000)
        let tap = RecordingTap(file: file, fileFormat: fileFormat,
                               converter: try #require(Recorder.makeConverter(from: input, to: fileFormat)),
                               inputFormat: input, live: nil, onLevel: { _, _ in })
        tap.handle(RecorderTests.tone(Self.format(44_100, 2), seconds: 0.5))
        #expect(tap.length == 0)
        tap.handle(RecorderTests.tone(input, seconds: 0.5))
        #expect(tap.length > 0)
    }

    @Test("the real engine's notice is passed on, on the main thread")
    func systemEngineForwardsTheNotice() async {
        let engine = SystemEngine()
        var heard = 0
        engine.onConfigurationChange = { heard += 1 }
        NotificationCenter.default.post(name: .AVAudioEngineConfigurationChange,
                                        object: engine.engine)
        #expect(await eventually { heard == 1 })
        // Another engine's notice is not this one's.
        NotificationCenter.default.post(name: .AVAudioEngineConfigurationChange,
                                        object: AVAudioEngine())
        await idle(0.1)
        #expect(heard == 1)
    }
}
