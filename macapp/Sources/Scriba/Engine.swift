import Foundation

/// Bridge to the Python engine.
///
/// The app reimplements nothing: `scriba` does the work, this class launches it,
/// reads its output to follow where it has got to, and re-reads the state once it is done.
/// That is deliberate. The engine has to stay usable on its own from the terminal,
/// and an app that duplicates the logic starts drifting the day after.
/// The part of a run that changes several times a second.
///
/// Kept apart from the Engine on purpose. When these lived on the Engine, every
/// publish rebuilt every view observing it, which was the whole window: the
/// sidebar list, the toolbar and whatever was in the detail pane, four times a
/// second for the length of a transcription. Only the two views that draw
/// progress observe this one.
@MainActor
final class RunState: ObservableObject {
    @Published var phase: Phase = .idle
    @Published var lastLine = ""
    @Published var progress: Double?

    /// A new run, or none. The bar and the last line go with the phase: kept,
    /// the second recording of a batch opened at 100% under "Preparing the
    /// audio", with the first one's final line under it.
    func reset(to phase: Phase) {
        self.phase = phase
        progress = nil
        lastLine = ""
    }
}

@MainActor
final class Engine: ObservableObject {

    /// Live progress, observed only where it is drawn.
    let live = RunState()

    /// Where the run has got to. Published because the queue and the panels
    /// switch on it, and it changes seven times in a run rather than hundreds.
    @Published var phase: Phase = .idle

    /// The newest line worth showing, and how far the current stage has got.
    ///
    /// The whole log used to be @Published and appended to per line. whisper
    /// prints a progress percentage continuously, so every one of those lines
    /// republished the entire array and SwiftUI rebuilt everything observing this
    /// object, sidebar included. The window went sticky for the length of a
    /// transcription, which is exactly as long as somebody most wants to use it.
    /// Kept for the failure message. Not published: nothing draws from it while
    /// the job runs, and republishing it was the whole problem.
    private(set) var log: [String] = []
    private var lastPublish = Date.distantPast
    @Published var info: JobInfo?
    @Published var speakers: [Speaker] = []
    @Published var errorText: String?
    @Published var isRunning = false
    /// True while the state is being read in the background, so the interface can
    /// say it is busy instead of appearing frozen.
    @Published var isLoadingState = false

    private var process: Process?

    /// Set while a stop the user asked for is in flight.
    ///
    /// Terminating the child makes it exit non-zero, which is indistinguishable
    /// from a crash unless somebody writes it down. Without this the Stop button
    /// worked and then apologised for working, with an alert quoting the last six
    /// lines of a log nobody needed.
    private(set) var wasCancelled = false

    /// What the engine itself said went wrong in the run that just ended: the
    /// line it printed after its marker, or the sentence for a Python that could
    /// not be started. Nil when the engine said nothing readable.
    ///
    /// Kept apart from `errorText` on purpose. `errorText` is what the alert
    /// shows, so the Close button empties it, and the marker arrives while a
    /// process holding twenty gigabytes of Metal memory is still exiting, which
    /// is time enough to press Close. The run then ended with nothing in
    /// `errorText`, and the row was given the tail of the log, whose last line
    /// was the end of a wrapped sentence about memory allocations.
    private(set) var failureReason: String?

    /// One line saying why the last run stopped, for a list that has room for one
    /// line. The alert carries the full text and is dismissed once; this stays.
    var failureSummary: String { Engine.summarise(failureReason ?? errorText) }

    /// A message of several lines on one, for a place with room for one.
    nonisolated static func oneLine(_ text: String) -> String {
        text.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    /// The one line worth keeping out of an error text.
    ///
    /// When the engine could be understood, the text is its own sentence and
    /// the first line is it. When it could not, the text is "Exit code N." over
    /// the tail of the log, and the first line said nothing: every failed row
    /// read "Exit code 1." The engine prints the reason last (cli.py, `run`),
    /// so the last line that is not the exit code and not the tally is the one.
    nonisolated static func summarise(_ text: String?) -> String {
        guard let text, !text.isEmpty else { return "did not finish" }
        let lines = text.split(separator: "\n")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        guard let first = lines.first else { return "did not finish" }
        var chosen = first
        if first.hasPrefix("Exit code") {
            let telling = lines.dropFirst().filter {
                !$0.hasPrefix("Exit code") && !$0.hasSuffix("did not go through:")
                    && !$0.hasPrefix("─")
            }
            chosen = telling.last ?? first
        }
        return chosen.count > 90 ? String(chosen.prefix(88)) + "…" : chosen
    }

    // MARK: - configuration

    /// Where the Python interpreter with scriba installed lives.
    ///
    /// The guess is the conda environment holding whisperX 3.8.6 and pyannote 4, the
    /// combination that puts diarization on Metal. Settings overrides it, and the
    /// error path names the path it tried, because "nothing happens when I press
    /// Transcribe" is the least useful thing an app can do.
    static var pythonPath: String {
        get {
            UserDefaults.standard.string(forKey: "pythonPath")
                ?? "/opt/homebrew/Caskroom/miniforge/base/envs/whisperx4/bin/python"
        }
        set { UserDefaults.standard.set(newValue, forKey: "pythonPath") }
    }

    /// Folder of the Python package (only needed when scriba is not pip-installed).
    static var enginePath: String {
        get {
            UserDefaults.standard.string(forKey: "enginePath")
                ?? (NSHomeDirectory() as NSString).appendingPathComponent("dev/scriba")
        }
        set { UserDefaults.standard.set(newValue, forKey: "enginePath") }
    }

    static var isConfigured: Bool {
        FileManager.default.isExecutableFile(atPath: pythonPath)
    }

    /// Whether the working directory we are about to hand the process exists.
    ///
    /// It does not have to hold the package: a pip install puts scriba on the
    /// path and the folder is then only a place to start from. It does have to
    /// exist, because Process throws before running anything if it does not, and
    /// the error that surfaces then names the interpreter and not the folder.
    static var isEngineFolderUsable: Bool {
        var isDir: ObjCBool = false
        let there = FileManager.default.fileExists(atPath: enginePath, isDirectory: &isDir)
        return there && isDir.boolValue
    }

    /// Mirrors ERR_NO_TOKEN in scriba/pipeline.py. The one string that genuinely
    /// crosses the Python/Swift boundary, so it is spelled out in one place on
    /// each side rather than buried inside a condition.
    nonisolated static let errNoToken = "[scriba:error:hf-token]"
    /// Mirrors ERR_MARK. Printed by the engine once per failed file, on its own
    /// line, when SCRIBA_MARKERS is set in its environment, which `launch`
    /// does. The line carries the reason, so the row can show the reason
    /// rather than the exit code.
    nonisolated static let errMark = "[scriba:error]"

    // MARK: - execution

    /// How a run ended. A queue needs all three: it moves on after the first two
    /// and stops after the third, and a stop the user asked for is not a failure.
    enum Outcome { case finished, failed, cancelled }

    /// `onFinish` receives the outcome, so a queue can move on to the next
    /// recording. Without it the caller has to poll `isRunning`, and a failure
    /// looks exactly like a success that ran quickly.
    func run(file: URL, language: String, minSpeakers: Int?, maxSpeakers: Int?,
             collection: String = "", onFinish: ((Outcome) -> Void)? = nil) {
        guard !isRunning else { return }
        var args = ["-m", "scriba.cli", "run", file.path, "--lang", language]
        if let m = minSpeakers { args += ["--min-speakers", String(m)] }
        if let m = maxSpeakers { args += ["--max-speakers", String(m)] }
        // The folder a recording came from has to outlive the queue: once it is
        // a job, nothing else groups forty files all called "Recording 12".
        if !collection.isEmpty { args += ["--collection", collection] }
        launch(args, on: file, startingPhase: .preparing, onFinish: onFinish)
    }

    /// True while `scriba name` is running, as opposed to a transcription.
    ///
    /// The two are different jobs of work and the interface said the same thing
    /// about both: pressing Save showed a progress strip with an empty filename
    /// walking through "preparing the audio", which is not what saving a name
    /// does and not how long it takes.
    @Published private(set) var isNaming = false

    func applyNames(file: URL, mapping: [String: String], enroll: Bool,
                    onFinish: ((Outcome) -> Void)? = nil) {
        guard !isRunning else { return }
        var args = ["-m", "scriba.cli", "name", file.path]
        args += Self.nameArguments(mapping)
        if !enroll { args.append("--no-enroll") }
        isNaming = true
        launch(args, on: file, startingPhase: .identifying) { [weak self] outcome in
            self?.isNaming = false
            onFinish?(outcome)
        }
    }

    /// `LABEL=Name` for each entry, in a fixed order. An empty name is sent as it
    /// is, `LABEL=`: that is how the engine is told to forget a name. Empty
    /// entries used to be dropped before the command was built, so a wrong name
    /// could be replaced but never removed.
    nonisolated static func nameArguments(_ mapping: [String: String]) -> [String] {
        mapping.sorted { $0.key < $1.key }.map {
            "\($0.key)=\($0.value.trimmingCharacters(in: .whitespacesAndNewlines))"
        }
    }

    /// What Save sends: every name typed, and an empty one for each voice whose
    /// stored name was wiped from its field. A voice with no stored name and an
    /// empty field has nothing to remove and is left out.
    nonisolated static func namesToSend(drafts: [String: String],
                                        stored: [String: String]) -> [String: String] {
        var send: [String: String] = [:]
        for (label, draft) in drafts {
            let name = draft.trimmingCharacters(in: .whitespacesAndNewlines)
            if !name.isEmpty {
                send[label] = name
            } else if !(stored[label] ?? "").isEmpty {
                send[label] = ""
            }
        }
        return send
    }

    private func launch(_ args: [String], on file: URL, startingPhase: Phase,
                        onFinish: ((Outcome) -> Void)? = nil) {
        isRunning = true
        wasCancelled = false
        errorText = nil
        failureReason = nil
        log.removeAll()
        lastPublish = .distantPast
        phase = startingPhase
        live.reset(to: startingPhase)

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: Self.pythonPath)
        proc.arguments = args
        // Fall back to the home directory rather than a folder that is not there.
        // The default points at a clone, and somebody who installed with pip has
        // no clone, so the run failed with a message about the interpreter.
        proc.currentDirectoryURL = URL(fileURLWithPath:
            Self.isEngineFolderUsable ? Self.enginePath : NSHomeDirectory())

        var env = ProcessInfo.processInfo.environment
        env["PYTHONPATH"] = Self.enginePath
        env["PYTHONUNBUFFERED"] = "1"
        // Each audio library opens its own thread pool; without this cap they fight
        // over the same cores and end up slower, not faster.
        // An app launched from the Finder inherits a PATH with almost nothing on
        // it, and the engine shells out to ffmpeg and ffprobe. Without this the
        // first stage failed with "has no audio track", which is a confident,
        // specific and wrong thing to say about a file that plays fine, and it
        // only happened for people who did not start the app from a terminal.
        let extraPath = [URL(fileURLWithPath: Self.pythonPath).deletingLastPathComponent().path,
                         "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"]
        env["PATH"] = (extraPath + [env["PATH"] ?? ""]).joined(separator: ":")
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["SCRIBA_MARKERS"] = "1"
        env["OMP_NUM_THREADS"] = String(max(ProcessInfo.processInfo.activeProcessorCount / 2, 1))
        proc.environment = env

        let outPipe = Pipe(), errPipe = Pipe()
        proc.standardOutput = outPipe
        proc.standardError = errPipe

        // Both pipes, and both for two separate reasons.
        //
        // Progress: the engine prints its phase lines on stdout. Both rich's Console
        // and plain print() go there, and stderr stays empty on a clean run. Reading
        // only stderr meant Phase.from never saw a single line, and the progress panel
        // sat on its first phase for the whole job.
        //
        // Survival: an unread pipe fills up. The OS buffer is 64 KB, whisper's own
        // progress output is far more than that, and a child process whose stdout is
        // full blocks forever. Draining stdout is not about showing it. It is what
        // keeps a long transcription from hanging.
        //
        // One reader per pipe, each with its own bytes and its own half line. They
        // shared one buffer once: a line half written on stdout was glued in front
        // of the next thing to arrive on stderr, and the marker that carries the
        // reason a run failed, no longer at the start of its line, was not
        // recognised.
        let outReader = PipeReader(outPipe.fileHandleForReading)
        let errReader = PipeReader(errPipe.fileHandleForReading)
        for reader in [outReader, errReader] {
            reader.handle.readabilityHandler = { [weak self] handle in
                let (lines, ended) = reader.readLines()
                // Nothing more will come, and a handler left in place is called
                // again at once, for ever, until somebody removes it.
                if ended { handle.readabilityHandler = nil }
                guard !lines.isEmpty else { return }
                Task { @MainActor in self?.ingest(lines) }
            }
        }

        proc.terminationHandler = { [weak self] p in
            // Whatever arrived after the last callback, and the line that had no
            // newline yet. The reason a run failed is usually in the final few
            // lines, and they were being dropped.
            outPipe.fileHandleForReading.readabilityHandler = nil
            errPipe.fileHandleForReading.readabilityHandler = nil
            let tail = outReader.finish() + errReader.finish()
            Task { @MainActor in
                // Stop, then start again straight away, and this handler belongs
                // to the process that was killed. Letting it run marked the new
                // run as failed and cleared the state out from under it; letting
                // it read its tail first wrote the killed run's last line into
                // the new run's progress.
                guard self?.process == nil || self?.process === p else { return }
                self?.ingest(tail)
                self?.isRunning = false
                self?.process = nil
                if self?.wasCancelled == true {
                    self?.wasCancelled = false
                    self?.phase = .idle
                    self?.live.reset(to: .idle)
                    self?.errorText = nil
                    self?.failureReason = nil
                    onFinish?(.cancelled)
                    return
                }
                let ok = p.terminationStatus == 0
                if ok {
                    self?.phase = .done
                    self?.reload(file: file)
                } else {
                    self?.phase = .failed
                    // Always something. The old version tested for nil, and an
                    // empty log produced an empty string, so the alert came up
                    // with a title, a Close button and no text at all. Not when
                    // the engine gave its reason: the alert was up with it, and
                    // closing it is not a reason to raise a worse one now.
                    if self?.failureReason == nil, (self?.errorText ?? "").isEmpty {
                        let tail = self?.log.suffix(6).joined(separator: "\n") ?? ""
                        self?.errorText = tail.isEmpty
                            ? "The engine exited with code \(p.terminationStatus) and said nothing."
                            : "Exit code \(p.terminationStatus).\n\n" + tail
                    }
                }
                onFinish?(ok ? .finished : .failed)
            }
        }

        do {
            try proc.run()
            process = proc
        } catch {
            isRunning = false
            phase = .failed
            let why = "Cannot launch Python at \(Self.pythonPath): \(error.localizedDescription)"
            errorText = why
            failureReason = why
            onFinish?(.failed)
        }
    }

    func cancel() {
        guard isRunning else { return }
        wasCancelled = true
        process?.terminate()
        process = nil
        isRunning = false
        phase = .idle
        live.reset(to: .idle)
        errorText = nil
        failureReason = nil
    }

    /// Kill the child before the app goes away.
    ///
    /// A terminated app leaves its subprocess running: whisper carries on with
    /// the whole processor busy and no window left to explain what is using it.
    /// Somebody who quits an app has already said what they want.
    ///
    /// Counted as a stop the person asked for. The app can stay alive a moment
    /// longer while it closes a recording, and the run ending then looked like a
    /// failure: the queue marked the row failed and, in a batch, started the next
    /// recording with nobody left to stop it.
    func terminateChild() {
        guard let p = process else { return }
        wasCancelled = true
        p.terminate()
        process = nil
    }

    /// The reason a line carries, if it is the engine saying why a run failed.
    ///
    /// Matched against a marker the engine emits, not against its wording.
    /// The previous version looked for a phrase from the human message and
    /// stopped working the day that sentence was rephrased. It broke silently,
    /// on a path no test covers. Keep this in step with ERR_NO_TOKEN and
    /// ERR_MARK in pipeline.py. A missing token prints both markers on one
    /// line, the token one inside the error one, so it is checked first and
    /// both come out.
    nonisolated static func failureReason(inLine line: String) -> String? {
        let reason: String
        if line.contains(errNoToken) {
            reason = line.replacingOccurrences(of: errNoToken, with: "")
                .replacingOccurrences(of: errMark, with: "")
        } else if let mark = line.range(of: errMark) {
            // Not only at the start of the line: a progress bar redrawn with a
            // carriage return and no newline left the marker behind it.
            reason = String(line[mark.upperBound...])
        } else {
            return nil
        }
        let trimmed = reason.trimmingCharacters(in: .whitespaces)
        return trimmed.isEmpty ? nil : trimmed
    }

    /// Lines, whole ones, from either pipe. Cutting the bytes into lines is the
    /// reader's job (PipeReader), so a line arrives here complete and in one
    /// piece however the operating system happened to hand it over.
    private func ingest(_ lines: [String]) {
        var newestPhase: Phase?
        var newestLine: String?
        var newestProgress: Double?

        for line in lines {
            let clean = line.trimmingCharacters(in: .whitespaces)
            guard !clean.isEmpty else { continue }
            // Before the noise filter below: a reason that happens to quote a
            // library's warning must not be thrown away with the warnings.
            if let reason = Engine.failureReason(inLine: clean) {
                failureReason = reason
                errorText = reason
            }
            // The libraries underneath (torch, speechbrain, lightning) are noisy and print
            // warnings that have nothing to do with the user. Filtering them here instead
            // of showing them is the difference between a readable log and a wall of text.
            if clean.hasPrefix("INFO:") || clean.hasPrefix("WARNING:")
                || clean.contains("UserWarning") || clean.hasPrefix("warn")
                || clean.contains("Lightning automatically") { continue }
            log.append(clean)
            if let p = Phase.from(logLine: clean) { newestPhase = p }
            if let pct = Engine.percentage(in: clean) {
                newestProgress = pct
            } else {
                newestLine = clean
            }
        }
        if log.count > 400 { log.removeFirst(log.count - 400) }

        // A phase change is worth a redraw immediately. Everything else waits for
        // the next tick, because the interface cannot show more than a few frames
        // a second and the engine can produce hundreds of lines in one.
        let now = Date()
        let phaseChanged = newestPhase != nil && newestPhase != phase
        guard phaseChanged || now.timeIntervalSince(lastPublish) > 0.25 else { return }
        lastPublish = now
        if let p = newestPhase, p != phase {
            phase = p
            live.phase = p
            live.progress = nil
        }
        if let pct = newestProgress { live.progress = pct }
        if let line = newestLine { live.lastLine = line }
    }

    /// The percentage whisper prints while it decodes, if this line carries one.
    ///
    /// A bar that moves is the difference between "this is working" and "this has
    /// hung", and the engine has been printing the number all along.
    static func percentage(in line: String) -> Double? {
        // Anchored to what whisper prints. Matching any percentage anywhere meant
        // the confidence figures in the language line drove the bar: two seconds
        // in it read 100% under "Detecting the language", then fell back to
        // nothing, which reads as a bug rather than as progress.
        guard let range = line.range(of: #"Progress:\s*(\d{1,3}(\.\d+)?)%"#,
                                     options: .regularExpression)
        else { return nil }
        let digits = line[range].drop(while: { !$0.isNumber }).prefix(while: { $0.isNumber || $0 == "." })
        guard let value = Double(digits) else { return nil }
        return min(max(value / 100, 0), 1)
    }

    // MARK: - reading the state

    /// Read a job the app already knows the folder of, from the folder.
    ///
    /// The state is two JSON files sitting on disk. Asking the engine for them
    /// meant starting a Python interpreter and importing numpy to hand back
    /// something already written down: three seconds of it, measured, of which
    /// the reading itself was two tenths of a millisecond. Opening a recording
    /// felt broken because it was.
    ///
    /// This duplicates no logic. What must not be written twice is how a job is
    /// derived, and that still happens in exactly one place. Reading a file the
    /// engine wrote is not deriving anything, and if the shape ever changes the
    /// decoder fails loudly and the subprocess path below is still there.
    /// Which job the interface is currently showing. A read that finishes after
    /// the user has moved on, or after a background run has touched something
    /// else, must not land: it used to replace the panel and empty the name
    /// fields somebody was halfway through typing.
    private(set) var showing: String?

    func load(jobDir: String, source: String) {
        showing = jobDir
        let dir = URL(fileURLWithPath: jobDir)
        let fm = FileManager.default

        guard let stateData = fm.contents(atPath: dir.appendingPathComponent("state.json").path),
              let state = try? JSONDecoder().decode(JobState.self, from: stateData)
        else {
            // No readable state on disk. Falling through to the engine costs
            // three seconds and, for a job folder that never got that far, ends
            // in the same silence. Say so instead.
            isLoadingState = false
            info = nil
            speakers = []
            errorText = "This job has no readable state. It was probably "
                      + "interrupted before it wrote anything. Run it again."
            return
        }

        let turnsURL = dir.appendingPathComponent("turns.json")
        let turns = fm.contents(atPath: turnsURL.path)
            .flatMap { try? JSONDecoder().decode([Turn].self, from: $0) } ?? []

        // The engine's own formats, not everything that ended up in the folder.
        // Opening it in the Finder once put a .DS_Store in the list of produced
        // documents, with a button offering to reveal it.
        let written = Set(["md", "txt", "srt", "vtt", "json"])
        let outDir = dir.appendingPathComponent("output")
        let outputs = ((try? fm.contentsOfDirectory(atPath: outDir.path)) ?? [])
            .filter { !$0.hasPrefix(".") && written.contains(($0 as NSString).pathExtension) }
            .sorted()
            .map { outDir.appendingPathComponent($0).path }

        let info = JobInfo(
            jobDir: jobDir, source: source, turns: turns, outputs: outputs,
            dossier: dir.appendingPathComponent("who-is-who.md").path,
            audio: dir.appendingPathComponent("audio16k.wav").path,
            state: state)

        guard showing == jobDir else { return }
        isLoadingState = false
        self.info = info
        speakers = Self.buildSpeakers(from: info)
    }

    /// Read the job state, off the main thread.
    ///
    /// This used to run the subprocess and wait for it right here, on the main actor.
    /// Starting Python and importing numpy costs about four seconds, so dropping a
    /// file froze the whole window for four seconds with nothing on screen to say
    /// why, and the controls that appear once a file is loaded could not draw until
    /// it was over. Under load, with a transcription already running, it is longer.
    func reload(file: URL) {
        isLoadingState = true
        let python = Self.pythonPath
        let root = Self.enginePath

        Task.detached(priority: .userInitiated) {
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: python)
            proc.arguments = ["-m", "scriba.cli", "info", file.path]
            proc.currentDirectoryURL = URL(fileURLWithPath: root)
            var env = ProcessInfo.processInfo.environment
            env["PYTHONPATH"] = root
            proc.environment = env
            let pipe = Pipe()
            proc.standardOutput = pipe
            proc.standardError = FileHandle.nullDevice

            var parsed: JobInfo?
            var failure: String?
            do {
                try proc.run()
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                proc.waitUntilExit()
                parsed = try? JSONDecoder().decode(JobInfo.self, from: data)
            } catch {
                failure = "Cannot read the state: \(error.localizedDescription)"
            }

            let result = parsed
            let problem = failure
            await MainActor.run { [weak self] in
                guard let self else { return }
                self.isLoadingState = false
                if let problem { self.errorText = problem }
                guard let result else { return }
                // Same rule as load(jobDir:). A read that comes back after the
                // selection moved on belongs to a different recording.
                guard self.showing == nil || self.showing == result.jobDir else { return }
                self.info = result
                self.speakers = Self.buildSpeakers(from: result)
            }
        }
    }

    /// Builds the speaker table from the turns. For each speaker it keeps the longest
    /// turn aside: that is the one to replay when deciding whose voice it is.
    static func buildSpeakers(from info: JobInfo) -> [Speaker] {
        var seconds: [String: Double] = [:]
        var counts: [String: Int] = [:]
        var best: [String: Turn] = [:]

        for t in info.turns {
            guard let s = t.speaker else { continue }
            seconds[s, default: 0] += t.end - t.start
            counts[s, default: 0] += 1
            if let cur = best[s] {
                if t.text.count > cur.text.count { best[s] = t }
            } else {
                best[s] = t
            }
        }

        let names = info.state.names ?? [:]
        let matches = info.state.matches ?? [:]

        return seconds.keys.sorted().map { id in
            let quote = best[id]
            let match = matches[id]
            return Speaker(
                id: id,
                // Only a certain match fills the field. A borderline candidate stays a
                // suggestion to accept by hand: pre-filling it would get the uncertain
                // cases accepted out of inertia, which is exactly the wrong outcome.
                name: names[id] ?? match?.name ?? "",
                suggested: match?.name ?? match?.candidate,
                score: match?.score ?? 0,
                reason: match?.reason ?? "",
                speechSeconds: seconds[id] ?? 0,
                turnCount: counts[id] ?? 0,
                previewStart: quote?.start ?? 0,
                previewEnd: min((quote?.start ?? 0) + 12, quote?.end ?? 12),
                longestQuote: quote?.text ?? ""
            )
        }
        .sorted { $0.speechSeconds > $1.speechSeconds }
    }
}

// MARK: - reading the pipes

/// Bytes from one pipe, cut into the lines they hold.
///
/// A pipe hands over bytes, not lines, and not even whole characters: a read can
/// end in the middle of a three-byte character. The chunk used to be decoded as
/// it came and thrown away whole when that failed, taking with it whatever else
/// it held, which could be the line that says why the run failed. A newline byte
/// never occurs inside a multi-byte UTF-8 character, so cutting at newlines
/// first and decoding afterwards loses nothing. A carriage return ends a line as
/// well: a progress bar that redraws itself is a stream of them.
struct LineBuffer {
    private var pending = Data()

    /// A single line longer than this is a progress bar printing with no
    /// newline, so it is not held for ever.
    static let longestLine = 4096

    /// Takes the next bytes and returns the lines they completed.
    mutating func append(_ data: Data) -> [String] {
        pending.append(data)
        var lines: [String] = []
        var start = 0
        let bytes = [UInt8](pending)
        for (i, byte) in bytes.enumerated() where byte == 0x0A || byte == 0x0D {
            lines.append(Self.text(bytes[start..<i]))
            start = i + 1
        }
        var rest = Array(bytes[start...])
        if rest.count > Self.longestLine {
            let cut = Self.wholeCharacters(in: rest)
            lines.append(Self.text(rest[..<cut]))
            rest.removeFirst(cut)
        }
        pending = Data(rest)
        return lines
    }

    /// The line that never got its newline, at the end of the stream.
    mutating func finish() -> [String] {
        defer { pending = Data() }
        return pending.isEmpty ? [] : [Self.text([UInt8](pending)[...])]
    }

    /// Lossy on purpose: a byte that is not UTF-8 becomes a replacement
    /// character and the rest of the line survives.
    private static func text(_ bytes: ArraySlice<UInt8>) -> String {
        String(decoding: bytes, as: UTF8.self)
    }

    /// How many leading bytes end on a character boundary: a character cut by
    /// the end of the data stays behind for the next read.
    private static func wholeCharacters(in bytes: [UInt8]) -> Int {
        var end = bytes.count
        var tail = 0
        while end > 0, tail < 3, bytes[end - 1] & 0xC0 == 0x80 {
            end -= 1
            tail += 1
        }
        guard end > 0 else { return bytes.count }
        let lead = bytes[end - 1]
        let expected = lead >= 0xF0 ? 3 : lead >= 0xE0 ? 2 : lead >= 0xC0 ? 1 : 0
        return expected > tail ? end - 1 : bytes.count
    }
}

/// One pipe of the child process, read on whichever thread asks.
///
/// The read and the cutting into lines happen under one lock, so the termination
/// handler reading what is left cannot overtake a read that is still being
/// handed on, and the line a chunk ended in the middle of is completed by the
/// next chunk of the same pipe and no other.
final class PipeReader: @unchecked Sendable {
    let handle: FileHandle
    private let lock = NSLock()
    private var buffer = LineBuffer()

    init(_ handle: FileHandle) { self.handle = handle }

    /// What is there now, as whole lines, and whether the stream has ended.
    func readLines() -> (lines: [String], ended: Bool) {
        lock.lock(); defer { lock.unlock() }
        let chunk = handle.availableData
        return (buffer.append(chunk), chunk.isEmpty)
    }

    /// Everything left, down to the last line without a newline.
    func finish() -> [String] {
        lock.lock(); defer { lock.unlock() }
        let rest = (try? handle.readToEnd()).flatMap { $0 } ?? Data()
        return buffer.append(rest) + buffer.finish()
    }
}
