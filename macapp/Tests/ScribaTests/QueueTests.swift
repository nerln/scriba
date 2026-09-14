import Foundation
import Testing

@testable import Scriba

/// The queue's arithmetic and words, checked without a window.
///
/// The header once counted the running row and the failed rows as waiting, and
/// the estimate and the start button were written and never shown. Everything
/// here is the part of the queue a person reads, so it is the part worth
/// pinning down.
struct QueueTests {

    static func url(_ name: String) -> URL {
        URL(fileURLWithPath: "/recordings/\(name)")
    }

    static func item(_ name: String, _ state: QueueItem.State = .waiting,
                     minutes: Double? = nil) -> QueueItem {
        var item = QueueItem(url: url(name))
        item.state = state
        item.minutes = minutes
        return item
    }

    // MARK: header

    @Test("the header counts the waiting rows and only those")
    func headerCountsWaiting() {
        var q = Queue()
        q.items = [Self.item("a.m4a"), Self.item("b.m4a"), Self.item("c.m4a")]
        #expect(q.title == "Waiting to be transcribed (3)")
    }

    @Test("a running row is said to be running, not waiting")
    func headerWhileRunning() {
        var q = Queue()
        q.items = [Self.item("a.m4a", .running), Self.item("b.m4a"), Self.item("c.m4a")]
        #expect(q.title == "Transcribing, 2 waiting")
        q.items = [Self.item("a.m4a", .running)]
        #expect(q.title == "Transcribing")
    }

    @Test("failed rows are counted apart")
    func headerWithFailures() {
        var q = Queue()
        q.items = [Self.item("a.m4a", .failed("no audio track")), Self.item("b.m4a")]
        #expect(q.title == "Waiting to be transcribed (1) · 1 did not finish")
        q.items = [Self.item("a.m4a", .failed("x")), Self.item("b.m4a", .failed("y"))]
        #expect(q.title == "2 did not finish")
    }

    @Test("rows that just finished do not read as waiting")
    func headerAllFinished() {
        var q = Queue()
        q.items = [Self.item("a.m4a", .finished)]
        #expect(q.title == "Transcribed")
    }

    // MARK: the start button and the estimate

    @Test("the button names how many it starts, and is absent with nothing to start")
    func startTitle() {
        var q = Queue()
        #expect(q.startTitle == nil)
        q.items = [Self.item("a.m4a")]
        #expect(q.startTitle == "Transcribe this one")
        q.items.append(Self.item("b.m4a"))
        #expect(q.startTitle == "Transcribe all 2")
        q.items = [Self.item("a.m4a", .running), Self.item("b.m4a", .failed("x"))]
        #expect(q.startTitle == nil)
    }

    @Test("the estimate says the audio and the work, and the work is the smaller number")
    func estimate() {
        var q = Queue()
        #expect(q.estimate == "")
        q.items = [Self.item("a.m4a", minutes: 40), Self.item("b.m4a", minutes: 20),
                   Self.item("c.m4a", .failed("x"), minutes: 100)]
        #expect(q.estimate == "60 min of audio, up to about 30 min of work.")
        q.items = [Self.item("a.m4a")]
        #expect(q.estimate == "Takes up to half the length of the recordings.")
    }

    @Test("the progress panel promises a continuation only when there is one")
    func remainingText() {
        #expect(Queue.remainingText(0, continues: true) == nil)
        #expect(Queue.remainingText(2, continues: true) == "2 more after this one")
        #expect(Queue.remainingText(1, continues: false)?.hasPrefix("1 more waiting.") == true)
        #expect(Queue.remainingText(1, continues: false)?.contains("Transcribe all") == true)
    }

    @Test("the Transcribe button is live only when pressing it does something")
    func canStart() {
        #expect(Queue.canStart(.waiting, engineBusy: false))
        #expect(!Queue.canStart(.waiting, engineBusy: true))
        #expect(!Queue.canStart(.running, engineBusy: false))
        #expect(!Queue.canStart(.finished, engineBusy: false))
        #expect(Queue.canStart(.failed("x"), engineBusy: false))
    }

    // MARK: changing it

    @Test("adding the same file twice lists it once, however the path is spelled")
    func addDedupes() {
        var q = Queue()
        let first = q.add([(url: Self.url("a.m4a"), collection: "")])
        #expect(first.count == 1)
        let again = q.add([(url: URL(fileURLWithPath: "/recordings/./a.m4a"), collection: "x"),
                           (url: Self.url("b.m4a"), collection: "")])
        #expect(again.map { $0.url.lastPathComponent } == ["b.m4a"])
        #expect(q.items.count == 2)
    }

    @Test("the settings given at add time land on each row")
    func addCarriesSettings() {
        var q = Queue()
        q.add([(url: Self.url("a.m4a"), collection: "March")], language: "it", speakers: 3)
        #expect(q.items[0].language == "it")
        #expect(q.items[0].speakers == 3)
        #expect(q.items[0].collection == "March")
    }

    @Test("the running row cannot be removed; the others can")
    func removeSparesRunning() {
        var q = Queue()
        q.items = [Self.item("a.m4a", .running), Self.item("b.m4a")]
        q.remove(q.items[0].id)
        q.remove(q.items[1].id)
        #expect(q.items.map { $0.url.lastPathComponent } == ["a.m4a"])
    }

    @Test("moving a row to the front puts it first among the waiting, behind the running one")
    func moveToFront() {
        var q = Queue()
        q.items = [Self.item("run.m4a", .running), Self.item("a.m4a"), Self.item("b.m4a"),
                   Self.item("c.m4a")]
        q.moveToFront(q.items[3].id)
        #expect(q.items.map { $0.url.lastPathComponent } == ["run.m4a", "c.m4a", "a.m4a", "b.m4a"])
        #expect(q.nextToRun?.url.lastPathComponent == "c.m4a")
    }

    @Test("stopping puts the running row back to waiting and leaves the rest alone")
    func stopAll() {
        var q = Queue()
        q.items = [Self.item("a.m4a", .running), Self.item("b.m4a", .failed("x")),
                   Self.item("c.m4a", .finished)]
        q.stopAll()
        #expect(q.items[0].state == .waiting)
        #expect(q.items[1].state == .failed("x"))
        #expect(q.items[2].state == .finished)
    }

    // MARK: files that are not really there

    @Test("a file too small to hold audio is refused at the door, with the size in the reason")
    func placeholderIsFailedOnAdd() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("scriba-queue-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let stub = dir.appendingPathComponent("stub.m4a")
        FileManager.default.createFile(atPath: stub.path, contents: Data(count: 9 * 1024))
        let real = dir.appendingPathComponent("real.m4a")
        FileManager.default.createFile(atPath: real.path, contents: Data(count: 200 * 1024))

        var q = Queue()
        q.add([(url: stub, collection: ""), (url: real, collection: "")])
        guard case .failed(let why) = q.items[0].state else {
            Issue.record("the stub should have been marked failed"); return
        }
        #expect(why.contains("9 KB"))
        #expect(why.contains("no audio behind it"))
        #expect(q.items[1].state == .waiting)
        #expect(q.startTitle == "Transcribe this one")
    }

    @Test("a file that is not there is nobody's problem to describe")
    func missingFileHasNoProblem() {
        #expect(Queue.problem(with: Self.url("nowhere.m4a")) == nil)
    }

    // MARK: surviving a quit

    @Test("what is saved is the waiting and the running rows, with their settings")
    func saved() {
        var q = Queue()
        q.items = [Self.item("a.m4a", .running), Self.item("b.m4a"),
                   Self.item("c.m4a", .failed("x")), Self.item("d.m4a", .finished)]
        q.items[1].language = "es"
        q.items[1].speakers = 4
        q.items[1].collection = "Inbox"
        let saved = q.saved
        #expect(saved.map(\.path) == ["/recordings/a.m4a", "/recordings/b.m4a"])
        #expect(saved[1] == Queue.Saved(path: "/recordings/b.m4a", collection: "Inbox",
                                        language: "es", speakers: 4))
    }

    @Test("restoring brings back the files that still exist, all as waiting")
    func restore() {
        let saved = [Queue.Saved(path: "/recordings/a.m4a", language: "it", speakers: 2),
                     Queue.Saved(path: "/recordings/gone.m4a")]
        let q = Queue.restore(saved, exists: { $0.hasSuffix("a.m4a") })
        #expect(q.items.count == 1)
        #expect(q.items[0].state == .waiting)
        #expect(q.items[0].language == "it")
        #expect(q.items[0].speakers == 2)
    }

    @Test("saved rows round-trip through JSON")
    func savedRoundTrip() throws {
        let saved = [Queue.Saved(path: "/r/a.m4a", collection: "Inbox", language: "es", speakers: 3)]
        let data = try JSONEncoder().encode(saved)
        let back = try JSONDecoder().decode([Queue.Saved].self, from: data)
        #expect(back == saved)
    }
}

/// One row per recording, and the words around the list.
struct ListTests {

    static func job(_ path: String, state: String = "nothing") -> JobSummary {
        JobSummary(jobDir: "/jobs/\(URL(fileURLWithPath: path).lastPathComponent)",
                   source: URL(fileURLWithPath: path).lastPathComponent,
                   sourcePath: path, recorded: "", duration: 0, state: state,
                   names: [:], speakers: 0, sizeMb: 0, hasOutput: state == "done")
    }

    @Test("a job whose recording is in the queue is not drawn a second time")
    func queuedJobsAreHidden() {
        let jobs = [Self.job("/r/a.m4a"), Self.job("/r/b.m4a", state: "done")]
        let shown = visibleJobs(jobs, hidingQueued: [canonicalPath("/r/./a.m4a")])
        #expect(shown.map(\.sourcePath) == ["/r/b.m4a"])
        #expect(visibleJobs(jobs, hidingQueued: []).count == 2)
    }

    @Test("a job that never wrote its source down is never hidden by accident")
    func emptySourceStays() {
        let jobs = [Self.job("")]
        #expect(visibleJobs(jobs, hidingQueued: [""]).count == 1
                || visibleJobs(jobs, hidingQueued: []).count == 1)
        #expect(visibleJobs(jobs, hidingQueued: ["/r/a.m4a"]).count == 1)
    }

    @Test("the sentence under an empty list defers to the banner about the engine")
    func emptyText() {
        #expect(readySectionEmptyText(loading: false, problem: "no python", filter: "") == nil)
        #expect(readySectionEmptyText(loading: true, problem: nil, filter: "") == "Reading the list…")
        #expect(readySectionEmptyText(loading: false, problem: nil, filter: "")?
                    .hasPrefix("Nothing yet.") == true)
        #expect(readySectionEmptyText(loading: false, problem: nil, filter: "ada") == "Nothing matches ada.")
    }

    @Test("a short recording is not zero minutes long")
    func shortDurations() {
        #expect(shortDuration(45) == "45 s")
        #expect(shortDuration(2700) == "45 min")
        #expect(shortDuration(0) == "0 s")
    }

    @Test("the engine's states have words, and a run in progress is not a failure")
    func labels() {
        #expect(Self.job("/r/a.m4a", state: "running").label.hasPrefix("Being transcribed now"))
        #expect(Self.job("/r/a.m4a", state: "running").isRunningElsewhere)
        #expect(!Self.job("/r/a.m4a", state: "nothing").isRunningElsewhere)
        #expect(Self.job("/r/a.m4a", state: "done").isFinished)
    }

    @Test("the one-line failure is the reason, not the exit code")
    func failureSummary() {
        let text = "Exit code 1.\n\n──── a.m4a ────\n  a.m4a is not an audio or video file.\n"
                 + "1 of 1 did not go through:\n  a.m4a: a.m4a is not an audio or video file."
        let line = Engine.summarise(text)
        #expect(line.hasPrefix("a.m4a:"))
        #expect(!line.contains("Exit code"))
        #expect(Engine.summarise("Exit code 1.\n\nsomething") == "something")
        #expect(Engine.summarise("Exit code 2.") == "Exit code 2.")
        let token = "The Hugging Face token for pyannote is missing."
        #expect(Engine.summarise(token) == token)
        #expect(Engine.summarise(nil) == "did not finish")
        #expect(Engine.summarise(String(repeating: "x", count: 120)).count == 89)
    }
}

/// What the inbox offers, and what stays dismissed.
struct InboxTests {

    @Test("the inbox offers what has no job and was not swiped away")
    func waiting() {
        let present = [(url: URL(fileURLWithPath: "/in/a.m4a"), collection: "Inbox"),
                       (url: URL(fileURLWithPath: "/in/b.m4a"), collection: "Inbox"),
                       (url: URL(fileURLWithPath: "/in/c.m4a"), collection: "Inbox")]
        let offered = Inbox.waiting(among: present,
                                    known: [canonicalPath("/in/a.m4a")],
                                    dismissed: [canonicalPath("/in/b.m4a")])
        #expect(offered.map { $0.url.lastPathComponent } == ["c.m4a"])
        #expect(offered.allSatisfy { $0.collection == Inbox.collection })
    }

    @Test("a missing inbox folder is an empty inbox, not an error")
    func missingFolder() {
        let nowhere = URL(fileURLWithPath: "/nonexistent/scriba-inbox-\(UUID().uuidString)")
        #expect(Inbox.contents(in: nowhere).isEmpty)
    }

    @Test("the inbox lists recordings and skips what is not one")
    func contents() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("scriba-inbox-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        for name in ["one.m4a", "two.mp3", "notes.txt", ".DS_Store"] {
            FileManager.default.createFile(atPath: dir.appendingPathComponent(name).path,
                                           contents: Data(count: 64))
        }
        let found = Inbox.contents(in: dir).map { $0.url.lastPathComponent }
        #expect(found == ["one.m4a", "two.mp3"])
    }

    @Test("dismissing remembers, pruning forgets what is gone")
    func dismissed() {
        let suite = "scriba-tests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        #expect(Dismissed.load(from: defaults).isEmpty)
        Dismissed.add("/in/./a.m4a", to: defaults)
        Dismissed.add("/in/b.m4a", to: defaults)
        #expect(Dismissed.load(from: defaults) == [canonicalPath("/in/a.m4a"), canonicalPath("/in/b.m4a")])
        Dismissed.prune(keeping: [canonicalPath("/in/b.m4a")], in: defaults)
        #expect(Dismissed.load(from: defaults) == [canonicalPath("/in/b.m4a")])
    }
}
