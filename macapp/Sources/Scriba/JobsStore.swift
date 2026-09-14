import Foundation

/// One recording scriba knows about, whether it has been processed or not.
struct JobSummary: Codable, Identifiable, Hashable {
    var id: String { jobDir }
    let jobDir: String
    let source: String
    let sourcePath: String
    let recorded: String
    let duration: Double
    let state: String
    let names: [String: String]
    let speakers: Int
    let sizeMb: Double
    let hasOutput: Bool
    /// Out of the everyday list, but every byte still on disk.
    var archived: Bool = false
    /// The subfolder it was found in, when it came from a folder rather than a
    /// single file. It is the only thing that says which recordings belong
    /// together when somebody adds a tree of them at once.
    var collection: String = ""
    /// Why the last run stopped, in the engine's words. Empty when it did not
    /// stop, or when the folder is from before the engine wrote this down.
    var failed: String = ""

    enum CodingKeys: String, CodingKey {
        case jobDir = "job_dir"
        case source
        case sourcePath = "source_path"
        case recorded, duration, state, names, speakers
        case sizeMb = "size_mb"
        case hasOutput = "has_output"
        case archived, collection, failed
    }

    /// What to show in the list. The engine's words are for a terminal.
    var label: String {
        switch state {
        case "done":        return "Ready to read"
        case "running":     return "Being transcribed now, outside this window"
        case "transcribed": return "Transcribed, no document written"
        case "voices only": return "Voices separated, words missing"
        case "text only":   return "Words only, voices not separated"
        default:            return "Started and produced nothing"
        }
    }

    var isFinished: Bool { state == "done" }

    /// A run somebody else started, from a terminal or another copy of the app,
    /// is on it this minute. The engine leaves a note in the folder while it
    /// works; without it this row read as a failure for as long as the run took.
    var isRunningElsewhere: Bool { state == "running" }
}

/// The job rows to draw, given which recordings are in the queue right now.
///
/// A recording that is queued is already one row, under "Waiting". The job
/// folder the engine makes for it exists from the first second of the run, so
/// the list would show the same file twice, with two different descriptions,
/// after a Stop, after a failure, or when the list was refreshed mid-run.
func visibleJobs(_ jobs: [JobSummary], hidingQueued paths: Set<String>) -> [JobSummary] {
    jobs.filter { !paths.contains(canonicalPath($0.sourcePath)) }
}

/// One spelling for a path, so the queue and the engine agree on which file
/// they are talking about. The engine resolves symlinks before it writes the
/// path down (pipeline.py, Job.__init__); the queue holds whatever the Finder
/// handed over.
func canonicalPath(_ path: String) -> String {
    guard !path.isEmpty else { return path }
    return URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath().path
}

/// A recording waiting to be processed, or being processed now.
struct QueueItem: Identifiable, Hashable {
    enum State: Hashable {
        case waiting
        case running
        case finished
        case failed(String)

        var isFailed: Bool {
            if case .failed = self { return true }
            return false
        }
    }
    let id = UUID()
    let url: URL
    /// Where it sat relative to the folder it was added from, empty for a file
    /// added on its own.
    var collection: String = ""
    var state: State = .waiting
    /// Read once when the file joins the queue. Reading it from the estimate
    /// meant opening every queued file's container on the main thread on every
    /// redraw, which is why dropping a batch of recordings froze the window.
    var minutes: Double?
    /// The two settings that change what comes out, kept with the recording
    /// they were chosen for. They used to be one pair of values for the whole
    /// window, so picking Italian for one file quietly picked it for the next.
    var language: String = "auto"
    /// How many people were in the room, 0 for "as many as it finds".
    var speakers: Int = 0

    /// The file this item stands for, spelled the way the engine spells it.
    var path: String { canonicalPath(url.path) }
}

/// The list of everything scriba has touched.
///
/// Read through `scriba jobs list --json` rather than by poking at the job folder
/// from Swift. One source of truth for what a job is, and the CLI stays the thing
/// that defines it.
@MainActor
final class JobsStore: ObservableObject {
    @Published var jobs: [JobSummary] = []
    @Published var isLoading = false
    /// Set when the engine could not be asked. The difference between "you have
    /// no recordings" and "I could not find out" matters: the second one used to
    /// be shown as the first, on a machine holding dozens of transcripts.
    @Published var problem: String?

    /// For callers that need the list to be current before they act on it.
    func reloadAndWait() async {
        reload()
        while isLoading { try? await Task.sleep(nanoseconds: 40_000_000) }
    }

    func reload() {
        // Reading the list starts a Python interpreter, which takes about three
        // seconds. Command-tabbing in and out a few times used to put three or
        // four of them on the processor at once, each one racing to publish.
        guard !isLoading else { return }
        isLoading = true
        let python = Engine.pythonPath
        let root = Engine.enginePath

        Task.detached(priority: .utility) {
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: python)
            proc.arguments = ["-m", "scriba.cli", "jobs", "list", "--json"]
            proc.currentDirectoryURL = URL(fileURLWithPath: root)
            var env = ProcessInfo.processInfo.environment
            env["PYTHONPATH"] = root
            proc.environment = env
            let pipe = Pipe()
            proc.standardOutput = pipe
            proc.standardError = FileHandle.nullDevice

            var loaded: [JobSummary]?
            var failure: String?
            if (try? proc.run()) != nil {
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                proc.waitUntilExit()
                if proc.terminationStatus != 0 {
                    failure = "The engine exited with code \(proc.terminationStatus) "
                            + "when asked for the list of recordings."
                } else {
                    loaded = try? JSONDecoder().decode([JobSummary].self, from: data)
                    if loaded == nil { failure = "The list of recordings could not be read." }
                }
            } else {
                failure = "Could not run \(python). Open Settings and check the path."
            }

            let result = loaded
            let problem = failure
            await MainActor.run { [weak self] in
                guard let self else { return }
                self.isLoading = false
                self.problem = problem
                // Only replace the list with one that was actually read. A failed
                // read used to empty the sidebar, and the app looked like a fresh
                // install with nothing in it.
                if let result, result != self.jobs { self.jobs = result }
            }
        }
    }

    /// Take a recording out of the everyday list, or put it back.
    ///
    /// The list moves straight away and the engine is told afterwards. Waiting
    /// for a Python interpreter to start means a row that sits there for a
    /// second after the swipe, which reads as a swipe that did not work.
    func archive(jobDir: String, value: Bool) async {
        if let i = jobs.firstIndex(where: { $0.jobDir == jobDir }) {
            jobs[i].archived = value
        }
        await run(["jobs", "archive", jobDir] + (value ? [] : ["--undo"]))
    }

    /// Delete a job folder, and the recording it came from only if asked.
    func forget(jobDir: String, withSource: Bool) async {
        jobs.removeAll { $0.jobDir == jobDir }
        await run(["jobs", "forget", jobDir, "--yes"] + (withSource ? ["--with-source"] : []))
        reload()
    }

    /// One engine command, waited for, output discarded.
    ///
    /// A failure reloads the list rather than raising a dialog: the sidebar is
    /// the truth, so a row that comes back says the archive did not happen, and
    /// says it where the person is already looking.
    private func run(_ arguments: [String]) async {
        let python = Engine.pythonPath
        let root = Engine.enginePath
        let ok = await Task.detached(priority: .userInitiated) { () -> Bool in
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: python)
            proc.arguments = ["-m", "scriba.cli"] + arguments
            proc.currentDirectoryURL = URL(fileURLWithPath: root)
            var env = ProcessInfo.processInfo.environment
            env["PYTHONPATH"] = root
            proc.environment = env
            proc.standardOutput = FileHandle.nullDevice
            proc.standardError = FileHandle.nullDevice
            guard (try? proc.run()) != nil else { return false }
            proc.waitUntilExit()
            return proc.terminationStatus == 0
        }.value
        if !ok { reload() }
    }
}
