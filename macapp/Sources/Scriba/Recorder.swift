import AVFoundation
import Combine
import Foundation
import os
import SwiftUI

/// Recording a conversation straight into scriba.
///
/// Until now the path from a conversation to a document went through the phone,
/// Voice Memos, an export to the Desktop and a drag into this window. The audio
/// this produces is an ordinary file in an ordinary folder, and once it is
/// stopped it goes through the same pipeline as anything dropped in: whisper,
/// alignment, pyannote, the voice registry. Nothing here shortcuts that.
///
/// Three objects rather than one, and the split is not cosmetic. `Recorder`
/// changes state a handful of times per session. `LiveMeter` changes about ten
/// times a second, and `LiveText` changes whenever the speech model has another
/// guess. SwiftUI redraws every view observing an object that changed, so
/// putting the level meter on the same object as everything else would redraw
/// the whole window ten times a second. Engine.swift carries a comment about
/// this exact mistake having been made once already.

// MARK: - level and elapsed time

/// The two numbers that move constantly while recording.
@MainActor final class LiveMeter: ObservableObject {
    @Published private(set) var peak: Float = 0
    @Published private(set) var seconds: Double = 0

    /// Written from the audio thread's callback, so it is coalesced: the tap
    /// fires about every 100 ms and the eye cannot read faster than that anyway.
    private var lastPublish = Date.distantPast

    func update(peak: Float, seconds: Double) {
        let now = Date()
        guard now.timeIntervalSince(lastPublish) >= 0.1 else { return }
        lastPublish = now
        self.peak = peak
        self.seconds = seconds
    }

    func reset() {
        peak = 0
        seconds = 0
        lastPublish = .distantPast
    }
}

/// What the live speech model thinks it has heard so far.
///
/// `settled` is what it has committed to; `guess` is the hypothesis for the
/// words being spoken right now, which it will revise. They are kept apart
/// because the interface shows them differently, and because a reader deserves
/// to know which half is still moving.
@MainActor final class LiveText: ObservableObject {
    @Published private(set) var settled: String = ""
    @Published private(set) var guess: String = ""
    @Published private(set) var unavailable: String?

    private var lastGuessPublish = Date.distantPast

    func commit(_ text: String) {
        let clean = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clean.isEmpty else { return }
        settled = settled.isEmpty ? clean : settled + " " + clean
        guess = ""
        lastGuessPublish = .distantPast
    }

    func hypothesise(_ text: String) {
        let now = Date()
        guard now.timeIntervalSince(lastGuessPublish) >= 0.1 else { return }
        lastGuessPublish = now
        guess = text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func note(unavailable reason: String?) { unavailable = reason }

    func reset() {
        settled = ""
        guess = ""
        unavailable = nil
        lastGuessPublish = .distantPast
    }
}

// MARK: - the engine, and what the tap does

/// The audio engine, as far as the recorder needs it.
///
/// A seam, and only a seam: the real one is `SystemEngine` and the recorder uses
/// nothing else in production. It exists so that a change of input (AirPods
/// connecting, a USB microphone pulled out) can be played to the recorder in a
/// test, where nobody is there to plug anything in.
protocol RecordingEngine: AnyObject {
    /// What the input produces right now, or nil when there is no usable input:
    /// a sample rate or a channel count of zero is how the system says nothing is
    /// connected.
    var inputFormat: AVAudioFormat? { get }
    /// `format` is what the input produces, as read just before.
    func installTap(format: AVAudioFormat, block: @escaping AVAudioNodeTapBlock)
    func removeTap()
    func start() throws
    func stop()
    /// Called, on the main thread, when the system changes the input under a
    /// running engine. The engine has stopped by then and stays stopped until it
    /// is started again.
    var onConfigurationChange: (() -> Void)? { get set }
}

final class SystemEngine: RecordingEngine, @unchecked Sendable {
    let engine = AVAudioEngine()
    private var observer: NSObjectProtocol?
    var onConfigurationChange: (() -> Void)?

    init() {
        // The main queue, so the handler is read and written from one thread.
        observer = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main
        ) { [weak self] _ in self?.onConfigurationChange?() }
    }

    deinit {
        if let observer { NotificationCenter.default.removeObserver(observer) }
    }

    var inputFormat: AVAudioFormat? {
        let format = engine.inputNode.outputFormat(forBus: 0)
        return format.sampleRate > 0 && format.channelCount > 0 ? format : nil
    }

    func installTap(format: AVAudioFormat, block: @escaping AVAudioNodeTapBlock) {
        engine.inputNode.installTap(onBus: 0, bufferSize: 4800, format: format, block: block)
    }

    func removeTap() { engine.inputNode.removeTap(onBus: 0) }
    func start() throws { try engine.start() }
    func stop() { engine.stop() }
}

/// What happens to each buffer the microphone hands over, and the one thing that
/// has to change when the microphone does.
///
/// The tap used to capture the converter, the file and the live transcriber, so
/// there was nothing to change: when an input appeared or went away the engine
/// stopped, the tap stopped being called, and the recording went on, clock and
/// all, with nothing written. A new input means a new converter and a new tap on
/// the new format. It does not mean a new file: the file keeps the format it was
/// opened with, so one recording is one file whatever the microphone did.
final class RecordingTap: @unchecked Sendable {
    let fileFormat: AVAudioFormat
    private let onLevel: (Float, Double) -> Void

    /// What the audio thread reads for every buffer and the main thread changes
    /// when the input does. Behind a lock held for the length of a copy, never
    /// across the conversion or the write: the audio thread must not wait for the
    /// main actor, and `removeTap` does not promise that no buffer is still on its
    /// way when it returns.
    private struct Parts {
        var file: AVAudioFile?
        var converter: AVAudioConverter
        var inputFormat: AVAudioFormat
        var live: LiveTranscribing?
    }
    private let parts: OSAllocatedUnfairLock<Parts>

    init(file: AVAudioFile, fileFormat: AVAudioFormat, converter: AVAudioConverter,
         inputFormat: AVAudioFormat, live: LiveTranscribing?,
         onLevel: @escaping (Float, Double) -> Void) {
        self.fileFormat = fileFormat
        self.onLevel = onLevel
        parts = OSAllocatedUnfairLock(uncheckedState: Parts(
            file: file, converter: converter, inputFormat: inputFormat, live: live))
    }

    var live: LiveTranscribing? {
        get { parts.withLockUnchecked { $0.live } }
        set { parts.withLockUnchecked { $0.live = newValue } }
    }

    /// Frames written so far.
    var length: AVAudioFramePosition { parts.withLockUnchecked { $0.file?.length ?? 0 } }

    /// Runs on the audio thread. Nothing here may wait for the main actor or
    /// allocate unpredictably; the one lock is held on both sides only for a copy.
    func handle(_ buffer: AVAudioPCMBuffer) {
        let now = parts.withLockUnchecked { $0 }
        guard let file = now.file else { return }
        // A converter is built for one input format, and a buffer of another does
        // not raise: it traps in the audio layer and takes the app down with the
        // recording still open. After a change of input a buffer can only arrive
        // in the old format by a race, and one dropped buffer is the better end.
        guard buffer.format.sampleRate == now.inputFormat.sampleRate,
              buffer.format.channelCount == now.inputFormat.channelCount else { return }
        let loudest = Recorder.peak(buffer)
        if let converted = Recorder.convert(buffer, with: now.converter, to: fileFormat) {
            try? file.write(from: converted)
        }
        now.live?.feed(buffer)
        onLevel(loudest, Double(file.length) / fileFormat.sampleRate)
    }

    /// Convert from `format` from now on. False if the new input cannot be
    /// converted to what the file holds.
    func retarget(to format: AVAudioFormat) -> Bool {
        guard let next = Recorder.makeConverter(from: format, to: fileFormat) else { return false }
        parts.withLockUnchecked {
            $0.converter = next
            $0.inputFormat = format
        }
        return true
    }

    /// Let go of the file, which is what closes it and writes the real length
    /// into the header. Returns how long it was.
    func close() -> AVAudioFramePosition {
        parts.withLockUnchecked {
            let frames = $0.file?.length ?? 0
            $0.file = nil
            return frames
        }
    }
}

// MARK: - the recorder

@MainActor final class Recorder: ObservableObject {
    enum State: Equatable { case idle, recording, failed(String) }

    @Published private(set) var state: State = .idle
    /// Where the finished recording ended up, for the caller to hand to the engine.
    @Published private(set) var finished: URL?
    /// Set when a session produced silence: a common and otherwise invisible failure.
    @Published var warning: String?

    let meter = LiveMeter()
    let live = LiveText()

    /// Whether to show live text while recording. Off by default: it is a preview,
    /// it costs a language decision up front, and the document does not come from it.
    @AppStorage("liveTextEnabled") var liveTextEnabled = false

    nonisolated static let folder = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent(".scriba/recordings")

    /// The recording ended by itself: the microphone it was using went away and
    /// could not be started again on another. Carries the file, closed, or nil
    /// when there was too little to keep. `warning` says why. The model queues
    /// the file as it would after a Stop; nobody pressed anything, so nobody else
    /// is going to.
    let stoppedItself = PassthroughSubject<URL?, Never>()

    private var engine: RecordingEngine?
    private var tap: RecordingTap?
    private var url: URL?
    private var started: Date?
    private var framesWritten: AVAudioFramePosition = 0
    private var loudestSeen: Float = 0
    private var transcriber: LiveTranscribing?

    /// The stop that is under way, for a second caller to wait for. It used to be
    /// a flag that sent a second `stop()` away with nil, which was right about not
    /// stopping twice (the second would find the file already let go of and delete
    /// the recording as "under a second") and wrong about everything else: Quit
    /// asked for a stop while the live text was still finishing, got nil at once,
    /// answered the quit, and the app exited before the first stop had closed the
    /// file or queued it.
    private var stopTask: Task<URL?, Never>?
    private var stopping: Bool { stopTask != nil }
    private var followingInput = false

    /// How often the engine is started again after the input changed, and how
    /// long to wait between tries. A pair of AirPods is still switching profile
    /// for a moment after the system says it changed, and the first start can
    /// fail for that alone. These are not settings.
    var restartAttempts = 3
    var restartPause: Double = 0.3

    /// Where the recording goes, where the live text switch is kept, the engine
    /// that records it, whether the microphone may be used, and how live text is
    /// made. The defaults are the real ones; a test puts others in so that a
    /// change of input can be played without a microphone, a permission dialog
    /// or a speech model.
    private let directory: URL
    private let makeEngine: () -> RecordingEngine
    private let microphoneAllowed: () async -> Bool
    private let makeLive: (String, AVAudioFormat) async throws -> LiveTranscribing

    init(directory: URL = Recorder.folder,
         defaults: UserDefaults = .standard,
         makeEngine: @escaping () -> RecordingEngine = { SystemEngine() },
         microphoneAllowed: @escaping () async -> Bool = { await Recorder.microphoneGranted() },
         makeLive: @escaping (String, AVAudioFormat) async throws -> LiveTranscribing
             = Recorder.systemLive) {
        _liveTextEnabled = AppStorage(wrappedValue: false, "liveTextEnabled", store: defaults)
        self.directory = directory
        self.makeEngine = makeEngine
        self.microphoneAllowed = microphoneAllowed
        self.makeLive = makeLive
    }

    /// The speech model built into macOS 26, or the reason there is none.
    nonisolated static func systemLive(language: String,
                                       format: AVAudioFormat) async throws -> LiveTranscribing {
        guard #available(macOS 26, *) else { throw RecorderError.liveNeedsNewerSystem }
        return try await LiveTranscriber(language: language, from: format)
    }

    var isRecording: Bool { state == .recording }

    // MARK: starting

    /// Ask for the microphone, then start. `language` is only used by the live
    /// preview: the document's language is still decided by the pipeline, which
    /// samples the whole recording rather than guessing from the first seconds.
    func start(language: String) async {
        guard state != .recording else { return }
        finished = nil
        warning = nil
        meter.reset()
        live.reset()
        loudestSeen = 0
        framesWritten = 0

        // Ask before arming anything. Starting the engine without the permission
        // does not fail: it records frames of digital silence, and the first sign
        // of trouble is a transcript of nothing half an hour later.
        guard await microphoneAllowed() else {
            state = .failed("Scriba cannot use the microphone. Give it permission in "
                            + "System Settings > Privacy & Security > Microphone, then try again.")
            return
        }

        do {
            try await beginSession(language: language)
            state = .recording
            started = Date()
        } catch {
            state = .failed(Self.readable(error))
        }
    }

    static func microphoneGranted() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .audio)
        default: return false
        }
    }

    private func beginSession(language: String) async throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let engine = makeEngine()
        guard let tapFormat = engine.inputFormat else {
            throw RecorderError.noInput
        }

        // 48 kHz mono 16-bit on disk. The pipeline resamples to 16 kHz itself and
        // applies loudnorm and a high-pass on the way, so writing anything fancier
        // here would only be thrown away; writing anything smaller would throw away
        // information before that filter got to see it.
        guard let fileFormat = AVAudioFormat(commonFormat: .pcmFormatInt16,
                                             sampleRate: 48_000, channels: 1,
                                             interleaved: true),
              let fileConverter = Self.makeConverter(from: tapFormat, to: fileFormat) else {
            throw RecorderError.noConverter
        }

        let url = try Self.reserve(in: directory, stamp: Self.stampFormatter.string(from: Date()))
        // The name is taken from here on, so it is given back if the session
        // never begins: an empty file called "Recording" in the folder helps
        // nobody, and would make the next one "Recording 2".
        var begun = false
        defer { if !begun { try? FileManager.default.removeItem(at: url) } }
        let file = try AVAudioFile(forWriting: url, settings: fileFormat.settings,
                                   commonFormat: .pcmFormatInt16, interleaved: true)

        // The live preview is a second, independent consumer of the same tap. It
        // gets its own converter: the analyzer wants its own format, and feeding
        // one converter's output into a file expecting another format is a trap
        // door, not an error (it traps in the audio layer and takes the app down).
        var live: LiveTranscribing?
        if liveTextEnabled {
            do {
                let t = try await makeLive(language, tapFormat)
                t.onFinal = { [weak self] text in
                    Task { @MainActor in self?.live.commit(text) }
                }
                t.onGuess = { [weak self] text in
                    Task { @MainActor in self?.live.hypothesise(text) }
                }
                live = t
            } catch {
                // A missing speech model must not cost the recording. Say what
                // happened in the pane and keep the microphone running.
                self.live.note(unavailable: Self.readable(error))
            }
        }

        let meter = self.meter
        let tap = RecordingTap(file: file, fileFormat: fileFormat, converter: fileConverter,
                               inputFormat: tapFormat, live: live) { [weak self] loudest, seconds in
            Task { @MainActor [weak self] in
                meter.update(peak: loudest, seconds: seconds)
                self?.loudestSeen = max(self?.loudestSeen ?? 0, loudest)
            }
        }
        engine.installTap(format: tapFormat) { [tap] buffer, _ in tap.handle(buffer) }
        engine.onConfigurationChange = { [weak self] in
            Task { @MainActor in await self?.inputChanged() }
        }

        try engine.start()
        begun = true
        self.engine = engine
        self.tap = tap
        self.url = url
        self.transcriber = live
    }

    // MARK: the input changing

    /// The input changed under a running recording: AirPods connected, a USB
    /// microphone was plugged in or pulled out. The system stopped the engine
    /// before saying so, and it stays stopped until it is started again. Nothing
    /// restarted it, so the clock went on, the level meter went flat, and the file
    /// stopped growing with nothing on screen to say so: the person found out when
    /// the transcript came back empty.
    ///
    /// The tap goes back in on whatever the input is now, converting to the format
    /// the file already has, and the engine starts again. If it cannot, the
    /// recording stops cleanly, as a Stop would have stopped it, and `warning`
    /// says why.
    func inputChanged() async {
        // One at a time. A second notice that arrives while the first is waiting
        // between tries is answered by the next try, which reads the input afresh.
        guard state == .recording, !stopping, !followingInput,
              let engine, let tap else { return }
        followingInput = true
        defer { followingInput = false }

        var failure: Error = RecorderError.noInput
        for attempt in 1...max(restartAttempts, 1) {
            if attempt > 1 {
                try? await Task.sleep(nanoseconds: UInt64(restartPause * 1_000_000_000))
                // Stopped by hand while waiting.
                guard self.engine === engine, state == .recording, !stopping else { return }
            }
            do {
                try followInput(engine: engine, tap: tap)
                return
            } catch {
                failure = error
            }
        }
        await stopAfterLosingInput(failure)
    }

    private func followInput(engine: RecordingEngine, tap: RecordingTap) throws {
        engine.removeTap()
        // A Mac with no microphone left, or one that has not yet switched to the
        // next: no format, and the tries that follow give it time.
        guard let format = engine.inputFormat else { throw RecorderError.noInput }
        guard tap.retarget(to: format) else { throw RecorderError.noConverter }
        followLiveText(to: format, tap: tap)
        engine.installTap(format: format) { [tap] buffer, _ in tap.handle(buffer) }
        try engine.start()
    }

    /// The live text goes on if it can take the new format. If it cannot, only the
    /// live text stops, and the pane says so: the recording is what matters, and
    /// the live text is a preview.
    private func followLiveText(to format: AVAudioFormat, tap: RecordingTap) {
        guard let transcriber else { return }
        if transcriber.follow(format) { return }
        self.transcriber = nil
        tap.live = nil
        live.note(unavailable: "Live text stopped when the microphone changed, and could "
                             + "not follow the new one. The recording goes on.")
        Task { await transcriber.finish() }
    }

    private func stopAfterLosingInput(_ failure: Error) async {
        let url = await stop()
        // `stop` may have said something of its own (too short, silent).
        warning = Self.inputLostMessage(failure, kept: url != nil, alsoSaid: warning)
        stoppedItself.send(url)
    }

    /// What the person is told when the recording had to end because of the input.
    nonisolated static func inputLostMessage(_ error: Error, kept: Bool,
                                             alsoSaid: String?) -> String {
        var text: String
        if let e = error as? RecorderError, case .noInput = e {
            text = "The microphone was disconnected and there is no other input to switch to."
        } else {
            let why = readable(error).trimmingCharacters(in: CharacterSet(charactersIn: ". "))
            text = "The microphone changed and could not be started again (\(why))."
        }
        if kept { text += " The recording was stopped there, and what was recorded is kept." }
        if let alsoSaid { text += " " + alsoSaid }
        return text
    }

    // MARK: stopping

    /// Stop, close the file, and hand back where it is. Safe to call twice: a call
    /// made while a stop is under way waits for that stop and gets what it got,
    /// so when this returns the file is closed, whoever asked first.
    @discardableResult
    func stop() async -> URL? {
        if let stopTask { return await stopTask.value }
        guard state == .recording || engine != nil else { return nil }
        // Made here and finished by the task itself, so the slot is empty the
        // moment the stop is over, before any of the callers waiting resumes.
        let task = Task { [self] () -> URL? in
            let url = await finishRecording()
            stopTask = nil
            return url
        }
        stopTask = task
        return await task.value
    }

    private func finishRecording() async -> URL? {
        engine?.onConfigurationChange = nil
        engine?.removeTap()
        engine?.stop()
        engine = nil

        await transcriber?.finish()
        transcriber = nil

        let length = tap?.length ?? 0
        let rate = tap?.fileFormat.sampleRate ?? 48_000
        // Closing the file is what writes the WAV header's real length. Dropping
        // the reference is the documented way to do it, and quitting without
        // doing it leaves a file whose duration every tool reads as unknown.
        _ = tap?.close()
        tap = nil

        state = .idle
        let seconds = Double(length) / rate
        guard let url, seconds >= 1.0 else {
            if let url { try? FileManager.default.removeItem(at: url) }
            self.url = nil
            warning = "That recording was under a second, so it was not kept."
            return nil
        }

        if loudestSeen < 0.001 {
            // Permission can be granted and the input still be the wrong device,
            // muted, or unplugged. The file exists and contains nothing.
            warning = "The recording came out silent: nothing reached the microphone. "
                    + "Check the input device in System Settings > Sound."
        }

        self.url = nil
        finished = url
        return url
    }

    // MARK: helpers

    private static let stampFormatter: DateFormatter = {
        let f = DateFormatter()
        // Sortable, and legible in a Finder window a month later. Colons are not
        // allowed in a file name on this filesystem, hence the dashes. Down to the
        // second: the minute alone named a recording started a few seconds after
        // the last one the same, and the new file replaced the old.
        f.dateFormat = "yyyy-MM-dd HH-mm-ss"
        return f
    }()

    /// A file name nobody has used, taken in the same step that checks it.
    ///
    /// The second in the stamp is not enough: a recording stopped and started
    /// again inside one second, or a clock that was set back, would still give
    /// the same name, and `AVAudioFile(forWriting:)` replaces what it finds. The
    /// name is claimed by creating the file with O_EXCL, which fails if anything
    /// is there, and the next free " 2", " 3" is tried. An existing recording is
    /// never opened for writing.
    nonisolated static func reserve(in folder: URL, stamp: String) throws -> URL {
        for n in 1...10_000 {
            let name = n == 1 ? "Recording \(stamp).wav" : "Recording \(stamp) \(n).wav"
            let url = folder.appendingPathComponent(name)
            let fd = open(url.path, O_WRONLY | O_CREAT | O_EXCL, 0o644)
            if fd >= 0 {
                close(fd)
                return url
            }
            guard errno == EEXIST else {
                throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
            }
        }
        throw POSIXError(.EEXIST)
    }

    nonisolated static func peak(_ buffer: AVAudioPCMBuffer) -> Float {
        guard let channel = buffer.floatChannelData else { return 0 }
        var loudest: Float = 0
        for i in 0..<Int(buffer.frameLength) {
            loudest = max(loudest, abs(channel[0][i]))
        }
        return loudest
    }

    /// The converter from what the microphone gives to what the file holds, or
    /// nil if there is no such conversion (or no microphone to convert from).
    nonisolated static func makeConverter(from input: AVAudioFormat,
                                          to output: AVAudioFormat) -> AVAudioConverter? {
        guard input.sampleRate > 0, input.channelCount > 0,
              let converter = AVAudioConverter(from: input, to: output) else { return nil }
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        return converter
    }

    /// One converter for the whole session, never one per buffer: a fresh
    /// converter resets the resampler's filter state and the seam is audible.
    ///
    /// nonisolated because it is called from the audio thread. Hopping to the main
    /// actor to resample audio would be a glitch you could hear.
    nonisolated static func convert(_ input: AVAudioPCMBuffer, with converter: AVAudioConverter,
                        to format: AVAudioFormat) -> AVAudioPCMBuffer? {
        let ratio = format.sampleRate / input.format.sampleRate
        let capacity = AVAudioFrameCount(Double(input.frameLength) * ratio) + 1024
        guard let output = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else {
            return nil
        }
        var handedOver = false
        var error: NSError?
        let status = converter.convert(to: output, error: &error) { _, status in
            if handedOver { status.pointee = .noDataNow; return nil }
            handedOver = true
            status.pointee = .haveData
            return input
        }
        guard status != .error, output.frameLength > 0 else { return nil }
        return output
    }

    nonisolated private static func readable(_ error: Error) -> String {
        if let e = error as? RecorderError { return e.message }
        return (error as NSError).localizedDescription
    }
}

enum RecorderError: Error {
    case noInput
    case noConverter
    case liveNeedsNewerSystem

    var message: String {
        switch self {
        case .noInput:
            return "No audio input. Check the microphone in System Settings > Sound."
        case .noConverter:
            return "This machine's microphone format could not be converted to a "
                 + "recordable one. Please report the input device."
        case .liveNeedsNewerSystem:
            return "Live text needs macOS 26. Recording still works."
        }
    }
}

/// What the recorder needs from a live transcriber, so the recorder does not have
/// to be annotated for macOS 26 from top to bottom.
protocol LiveTranscribing: AnyObject {
    /// A finished utterance: the model will not revise this.
    var onFinal: ((String) -> Void)? { get set }
    /// The current hypothesis, which it will revise.
    var onGuess: ((String) -> Void)? { get set }
    /// Called from the audio thread with the tap's own buffer.
    func feed(_ buffer: AVAudioPCMBuffer)
    /// The input changed to `format`. Called with the tap removed, before the new
    /// one goes in. False if the transcriber cannot take it.
    func follow(_ format: AVAudioFormat) -> Bool
    func finish() async
}
