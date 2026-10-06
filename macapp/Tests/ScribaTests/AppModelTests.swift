import AVFoundation
import Foundation
import Testing

@testable import Scriba

/// A folder with the stand-in interpreter, a log of what it was asked, and a
/// private defaults domain, so nothing here touches the real saved queue.
///
/// Its list of jobs is one recording, "a", whose state lives in marker files:
/// `archive` and `forget` write them and `list` reads them. `source` and `failed`
/// files, when present, say which recording the job is for and why it stopped. The read takes its
/// snapshot first and prints after a pause, as the real one does, so a read that
/// began before a command describes the world before it.
@MainActor
final class Rig {
    let dir: URL
    let log: URL
    let defaults: UserDefaults
    private let suite = "scriba-model-\(UUID().uuidString)"
    private let hadPython = UserDefaults.standard.string(forKey: "pythonPath")

    init() {
        dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("scriba-model-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        log = dir.appendingPathComponent("runs.log")
        FileManager.default.createFile(atPath: log.path, contents: Data())
        let script = dir.appendingPathComponent("fake-python")
        let job = #"{"job_dir":"/jobs/a","source":"a.m4a","source_path":"'$S'","recorded":"","duration":0,"state":"done","names":{},"speakers":0,"size_mb":0,"has_output":true,"archived":'$A',"collection":"","failed":"'$F'"}"#
        // $1 is -m, $2 the module, $3 the command, $4 the recording.
        let text = """
        #!/bin/sh
        D='\(dir.path)'
        b=$(basename "$4")
        case "$3" in
          run)
            echo "run $b" >> '\(log.path)'
            case "$b" in
              slow*) exec sleep 5 ;;
              bad*)  sleep 0.2; echo "[scriba:error] $b is not audio"; sleep 0.4; exit 1 ;;
              quiet*) sleep 0.2; echo "something unrelated"; exit 1 ;;
              token*) echo "[scriba:error] [scriba:error:hf-token] Add a token."; sleep 0.2; exit 1 ;;
              split*) printf '[scriba:error] half'; sleep 0.15; echo "noise" >&2
                      sleep 0.15; echo ' of it'; sleep 0.2; exit 1 ;;
              *)     sleep 0.2; echo "Progress: 50%"; exit 0 ;;
            esac ;;
          name)
            echo "name $b" >> '\(log.path)'
            sleep 0.8; exit 0 ;;
          jobs)
            case "$4" in
              list)
                if [ -f "$D/archived" ]; then A=true; else A=false; fi
                if [ -f "$D/source" ]; then S=$(cat "$D/source"); else S=/r/a.m4a; fi
                if [ -f "$D/failed" ]; then F=$(cat "$D/failed"); else F=; fi
                if [ ! -f "$D/job" ] || [ -f "$D/forgotten" ]; then echo "[]"; exit 0; fi
                sleep 0.6
                echo '[\(job)]' ;;
              archive)
                sleep 0.3
                if [ "$6" = "--undo" ]; then rm -f "$D/archived"; else touch "$D/archived"; fi ;;
              forget)
                sleep 0.3; touch "$D/forgotten" ;;
            esac ;;
          *) exit 1 ;;
        esac
        """
        try? text.write(to: script, atomically: true, encoding: .utf8)
        try? FileManager.default.setAttributes([.posixPermissions: 0o755],
                                               ofItemAtPath: script.path)
        Engine.pythonPath = script.path
        defaults = UserDefaults(suiteName: suite)!
    }

    func cleanup() {
        if let hadPython { Engine.pythonPath = hadPython }
        else { UserDefaults.standard.removeObject(forKey: "pythonPath") }
        defaults.removePersistentDomain(forName: suite)
        try? FileManager.default.removeItem(at: dir)
    }

    /// What the model asks the power management and the notification centre,
    /// counted, and whether the window is in front. None of it reaches the system.
    let power = FakeAssertion()
    let notifier = FakeNotifier()
    var frontmost = true

    func model(recorder: Recorder? = nil) -> AppModel {
        AppModel(defaults: defaults, awake: power.awake(), notifier: notifier,
                 isFrontmost: { [unowned self] in self.frontmost }, recorder: recorder)
    }

    /// Make the job list hold the one recording, "a".
    func listAJob() {
        FileManager.default.createFile(atPath: dir.appendingPathComponent("job").path,
                                       contents: Data())
    }

    /// The job list's one recording is `source`, and it stopped with `failed`.
    func listAJob(source: String, failed: String) {
        try? source.write(to: dir.appendingPathComponent("source"), atomically: true, encoding: .utf8)
        try? failed.write(to: dir.appendingPathComponent("failed"), atomically: true, encoding: .utf8)
        listAJob()
    }

    /// A recording the queue accepts: big enough not to be called a stub.
    @discardableResult
    func add(_ name: String, to model: AppModel) -> QueueItem {
        let url = dir.appendingPathComponent(name)
        FileManager.default.createFile(atPath: url.path, contents: Data(count: 200 * 1024))
        return model.queue.add([(url: url, collection: "")])[0]
    }

    /// What the interpreter was asked to do, in order, by recording name.
    func asked() -> [String] {
        ((try? String(contentsOf: log, encoding: .utf8)) ?? "")
            .split(separator: "\n").map(String.init)
    }
}

/// Wait for a condition the main actor will make true.
@MainActor
func eventually(_ seconds: Double = 20, _ condition: () -> Bool) async -> Bool {
    let deadline = Date().addingTimeInterval(seconds)
    while !condition() {
        if Date() > deadline { return false }
        try? await Task.sleep(nanoseconds: 20_000_000)
    }
    return true
}

@MainActor
func idle(_ seconds: Double) async {
    try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
}

/// Both suites inside swap the interpreter the whole app uses, which is one
/// global (`Engine.pythonPath`), so they must not run beside each other.
@MainActor
@Suite(.serialized)
struct StandInInterpreterTests {

    /// Everything the window used to own and the window must not be needed for.
    ///
    /// The model is driven with no view at all, against a shell script that stands in
    /// for the Python interpreter: it logs which recording it was asked for, sleeps a
    /// moment, and exits well or badly by the name of the file. That is the same
    /// situation as a window that was closed mid-run, and it is the only way to see
    /// what the queue does when a run ends, since the queue's decisions are made in
    /// the callback of a child process.
    @MainActor
    struct AppModelTests {

        static func state(_ model: AppModel, _ name: String) -> QueueItem.State? {
            model.queue.items.first { $0.url.lastPathComponent == name }?.state
        }

        // MARK: asking for one to go next

        @Test("a recording put next runs after the current one, and then the queue stops")
        func putNextStopsBehindIt() async {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            let a = rig.add("a.m4a", to: model)
            let b = rig.add("b.m4a", to: model)
            rig.add("c.m4a", to: model)

            model.start(a)
            #expect(model.engine.isRunning)
            model.start(b)
            #expect(!model.runningAll)
            #expect(model.askedNext.map { $0.url.lastPathComponent } == ["b.m4a"])

            #expect(await eventually { rig.asked() == ["run a.m4a", "run b.m4a"]
                                       && !model.engine.isRunning && model.queue.items.count == 1 })
            await idle(0.5)
            #expect(rig.asked() == ["run a.m4a", "run b.m4a"])
            #expect(Self.state(model, "c.m4a") == .waiting)
            #expect(!model.engine.isRunning)
            #expect(model.askedNext.isEmpty)
        }

        @Test("several recordings put next run in the order they were asked for")
        func putNextKeepsOrder() async {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            let a = rig.add("a.m4a", to: model)
            rig.add("x.m4a", to: model)
            let c = rig.add("c.m4a", to: model)
            let d = rig.add("d.m4a", to: model)

            model.start(a)
            model.start(d)
            model.start(c)
            #expect(model.queue.shown.map { $0.url.lastPathComponent }
                    == ["a.m4a", "d.m4a", "c.m4a", "x.m4a"])
            #expect(await eventually { rig.asked().count == 3 && !model.engine.isRunning })
            await idle(0.5)
            #expect(rig.asked() == ["run a.m4a", "run d.m4a", "run c.m4a"])
            #expect(Self.state(model, "x.m4a") == .waiting)
        }

        @Test("Transcribe all still carries on through everything that waits")
        func transcribeAllCarriesOn() async {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            for name in ["a.m4a", "b.m4a", "c.m4a"] { rig.add(name, to: model) }
            model.startAll()
            #expect(model.runningAll)
            #expect(await eventually { model.queue.isEmpty && !model.runningAll })
            #expect(rig.asked() == ["run a.m4a", "run b.m4a", "run c.m4a"])
        }

        @Test("pressing Transcribe all while one runs lets the waiting ones follow")
        func transcribeAllMidRun() async {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            let a = rig.add("a.m4a", to: model)
            rig.add("b.m4a", to: model)
            model.start(a)
            model.startAll()
            #expect(model.runningAll)
            #expect(await eventually { model.queue.isEmpty && !model.runningAll })
            #expect(rig.asked() == ["run a.m4a", "run b.m4a"])
        }

        // MARK: names being saved

        @Test("Transcribe all during a save of names arms nothing")
        func transcribeAllDuringNaming() async {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            let a = rig.add("a.m4a", to: model)
            rig.add("b.m4a", to: model)
            rig.add("c.m4a", to: model)

            model.engine.applyNames(file: a.url, mapping: ["SPEAKER_00": "Ada"], enroll: true)
            #expect(model.engine.isNaming)
            model.startAll()
            #expect(!model.runningAll)
            #expect(model.queue.running == nil)
            #expect(model.start(a) == false)

            #expect(await eventually { !model.engine.isRunning })
            // Later, one recording on its own: it must not take the others with it.
            model.start(a)
            #expect(await eventually { !model.engine.isRunning && Self.state(model, "a.m4a") == nil })
            await idle(0.5)
            #expect(rig.asked() == ["name a.m4a", "run a.m4a"])
            #expect(Self.state(model, "b.m4a") == .waiting)
            #expect(Self.state(model, "c.m4a") == .waiting)
        }

        // MARK: failures in a batch

        @Test("a batch holds the alert back, keeps the reasons on the rows, and says it once at the end")
        func batchFailuresAreSummarised() async {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            for name in ["a.m4a", "bad1.m4a", "bad2.m4a", "d.m4a"] { rig.add(name, to: model) }

            model.startAll()
            var alertedMidBatch = false
            #expect(await eventually {
                if model.runningAll, model.alertText != nil { alertedMidBatch = true }
                return !model.runningAll && !model.engine.isRunning
            })
            #expect(!alertedMidBatch)
            #expect(rig.asked() == ["run a.m4a", "run bad1.m4a", "run bad2.m4a", "run d.m4a"])
            #expect(Self.state(model, "bad1.m4a") == .failed("bad1.m4a is not audio"))
            #expect(Self.state(model, "bad2.m4a") == .failed("bad2.m4a is not audio"))

            let alert = model.alertText ?? ""
            #expect(alert.hasPrefix("2 recordings did not go through:"))
            #expect(alert.contains("bad1.m4a: bad1.m4a is not audio"))
            #expect(alert.contains("bad2.m4a: bad2.m4a is not audio"))
            #expect(!alert.contains("a.m4a: "))
        }

        // MARK: failures in a run of recordings asked for next

        @Test("a failure followed by a recording put next holds its alert, and says it once at the end")
        func putNextAfterAFailure() async {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            let bad = rig.add("bad.m4a", to: model)
            let next = rig.add("b.m4a", to: model)
            model.start(bad)
            model.start(next)
            #expect(model.chaining)

            var alertedMidChain = false
            var heldSomething = false
            #expect(await eventually {
                if model.chaining {
                    if model.alertText != nil { alertedMidChain = true }
                    if model.engine.errorText != nil { heldSomething = true }
                }
                return !model.chaining && !model.engine.isRunning
            })
            #expect(!alertedMidChain)
            #expect(heldSomething)          // the engine did raise one, and it was kept back
            #expect(rig.asked() == ["run bad.m4a", "run b.m4a"])
            #expect(Self.state(model, "bad.m4a") == .failed("bad.m4a is not audio"))
            let alert = model.alertText ?? ""
            #expect(alert.hasPrefix("1 recording did not go through:"))
            #expect(alert.contains("bad.m4a: bad.m4a is not audio"))
        }

        @Test("the last of the recordings put next failing is told with the one before it, once")
        func putNextEndsInAFailure() async {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            let first = rig.add("bad1.m4a", to: model)
            let second = rig.add("bad2.m4a", to: model)
            model.start(first)
            model.start(second)

            var alertedMidChain = false
            #expect(await eventually {
                if model.chaining, model.alertText != nil { alertedMidChain = true }
                return !model.chaining && !model.engine.isRunning
            })
            #expect(!alertedMidChain)
            let alert = model.alertText ?? ""
            #expect(alert.hasPrefix("2 recordings did not go through:"))
            #expect(alert.contains("bad1.m4a: bad1.m4a is not audio"))
            #expect(alert.contains("bad2.m4a: bad2.m4a is not audio"))
        }

        @Test("a put-next run with nothing that failed ends with no alert at all")
        func putNextWithNoFailure() async {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            let a = rig.add("a.m4a", to: model)
            let b = rig.add("b.m4a", to: model)
            model.start(a)
            model.start(b)
            #expect(await eventually { !model.chaining && !model.engine.isRunning })
            #expect(model.alertText == nil)
            #expect(rig.asked() == ["run a.m4a", "run b.m4a"])
        }

        @Test("a recording put next and then removed leaves a failure to be told as a single one")
        func putNextRemoved() async {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            let bad = rig.add("bad.m4a", to: model)
            let next = rig.add("b.m4a", to: model)
            model.start(bad)
            model.start(next)
            model.dismiss(next)
            #expect(await eventually { !model.chaining && !model.engine.isRunning })
            #expect(rig.asked() == ["run bad.m4a"])
            #expect((model.alertText ?? "").hasPrefix("1 recording did not go through:"))
        }

        @Test("a failure with nothing put next still raises its alert at once")
        func failureAloneAlerts() async {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            let bad = rig.add("bad.m4a", to: model)
            model.start(bad)
            #expect(!model.chaining)
            #expect(await eventually { model.alertText != nil })
            #expect(model.alertText == "bad.m4a is not audio")
            // Let the run end before the rig goes: the model asks it a question then.
            #expect(await eventually { !model.engine.isRunning })
            #expect(model.alertText == "bad.m4a is not audio")
        }

        @Test("Stop ends the run of recordings put next and drops what it was counting")
        func stopEndsTheChain() async {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            let slow = rig.add("slow.m4a", to: model)
            let next = rig.add("b.m4a", to: model)
            model.start(slow)
            model.start(next)
            #expect(model.chaining)
            #expect(await eventually { rig.asked() == ["run slow.m4a"] })
            model.stopEverything()
            #expect(!model.chaining)
            #expect(model.askedNext.isEmpty)
            #expect(await eventually { !model.engine.isRunning })
            #expect(rig.asked() == ["run slow.m4a"])
            #expect(model.alertText == nil)
        }

        // MARK: the reason a run failed

        @Test("closing the alert while the engine is still exiting does not cost the row its reason")
        func reasonSurvivesClose() async {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            let bad = rig.add("bad.m4a", to: model)
            model.start(bad)
            #expect(await eventually { model.engine.errorText != nil })
            model.engine.errorText = nil        // the Close button
            #expect(await eventually { !model.engine.isRunning })
            #expect(Self.state(model, "bad.m4a") == .failed("bad.m4a is not audio"))
            #expect(model.engine.failureReason == "bad.m4a is not audio")
            // Not raised again behind the person's back.
            #expect(model.engine.errorText == nil)
        }

        @Test("a line half written on one pipe is not glued to what arrives on the other")
        func pipesKeepTheirOwnLines() async {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            let split = rig.add("split.m4a", to: model)
            model.start(split)
            #expect(await eventually { !model.engine.isRunning })
            #expect(Self.state(model, "split.m4a") == .failed("half of it"))
            #expect(model.engine.failureReason == "half of it")
        }

        @Test("a missing token still comes through both markers, as the sentence alone")
        func tokenReason() async {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            let token = rig.add("token.m4a", to: model)
            model.start(token)
            #expect(await eventually { !model.engine.isRunning })
            #expect(model.engine.failureReason == "Add a token.")
            #expect(Self.state(model, "token.m4a") == .failed("Add a token."))
        }

        @Test("when the engine said nothing readable, the reason it wrote in the job is used")
        func reasonFromTheJob() async {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            let quiet = rig.add("quiet.m4a", to: model)
            rig.listAJob(source: quiet.path,
                         failed: "could not load the model: MPS backend out of memory")
            model.start(quiet)
            #expect(await eventually { !model.engine.isRunning })
            #expect(model.engine.failureReason == nil)
            #expect(await eventually {
                Self.state(model, "quiet.m4a")
                    == .failed("could not load the model: MPS backend out of memory")
            })
        }

        @Test("a failure with nothing written anywhere keeps the tail of the log")
        func reasonFromTheLogTail() async {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            let quiet = rig.add("quiet.m4a", to: model)
            model.start(quiet)
            #expect(await eventually { !model.engine.isRunning })
            await idle(1.5)     // the job read finds no job for this file
            #expect(Self.state(model, "quiet.m4a") == .failed("something unrelated"))
        }

        @Test("a recording on its own still raises the alert when it fails")
        func singleFailureAlerts() async {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            let bad = rig.add("bad.m4a", to: model)
            model.start(bad)
            #expect(await eventually { !model.engine.isRunning })
            #expect(model.alertText == "bad.m4a is not audio")
            #expect(Self.state(model, "bad.m4a") == .failed("bad.m4a is not audio"))
        }

        @Test("the summary of an earlier batch is not carried into the next")
        func summaryIsPerBatch() async {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            for name in ["bad1.m4a", "b.m4a"] { rig.add(name, to: model) }
            model.startAll()
            #expect(await eventually { !model.runningAll && !model.engine.isRunning })
            #expect(model.alertText?.hasPrefix("1 recording did not go through:") == true)

            model.engine.errorText = nil
            model.queue.items.removeAll { $0.state.isFailed }
            rig.add("e.m4a", to: model)
            rig.add("f.m4a", to: model)
            model.startAll()
            #expect(await eventually { !model.runningAll && !model.engine.isRunning })
            #expect(model.alertText == nil)
        }

        @Test("stopping the batch drops its failures instead of reporting them later")
        func stopClearsTheTally() async {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            for name in ["bad1.m4a", "slow.m4a", "z.m4a"] { rig.add(name, to: model) }
            model.startAll()
            #expect(await eventually { rig.asked().contains("run slow.m4a") })
            model.stopEverything()
            #expect(await eventually { !model.engine.isRunning })
            #expect(model.alertText == nil)
            #expect(!model.runningAll)
            #expect(Self.state(model, "slow.m4a") == .waiting)
        }

        // MARK: quitting, and the window not being there

        @Test("quitting stops the run without calling it a failure or starting the next one")
        func quittingIsNotAFailure() async {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            rig.add("slow.m4a", to: model)
            rig.add("b.m4a", to: model)
            model.startAll()
            #expect(model.runningAll)
            #expect(await eventually { rig.asked() == ["run slow.m4a"] })

            model.engine.terminateChild()
            #expect(await eventually { !model.engine.isRunning })
            await idle(0.5)
            #expect(rig.asked() == ["run slow.m4a"])
            #expect(Self.state(model, "slow.m4a") == .waiting)
            #expect(Self.state(model, "b.m4a") == .waiting)
            #expect(model.engine.errorText == nil)
        }

        @Test("a run that ends with no window open still leaves the queue, and the saved list, right")
        func finishingWithoutAWindow() async {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            let a = rig.add("a.m4a", to: model)
            rig.add("b.m4a", to: model)
            model.start(a)
            #expect(await eventually { !model.engine.isRunning && model.queue.items.count == 1 })

            // What a window opened now would show, and what the next launch restores.
            #expect(model.queue.items.map { $0.url.lastPathComponent } == ["b.m4a"])
            let again = rig.model()
            #expect(again.queue.items.map { $0.url.lastPathComponent } == ["b.m4a"])
        }

        @Test("the queue is written by the model whenever it changes")
        func queueIsSavedByTheModel() {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            rig.add("a.m4a", to: model)
            rig.add("b.m4a", to: model)
            model.queue.items[1].language = "it"
            model.queue.items[1].speakers = 2

            let back = rig.model()
            #expect(back.queue.items.map { $0.url.lastPathComponent } == ["a.m4a", "b.m4a"])
            #expect(back.queue.items[1].language == "it")
            #expect(back.queue.items[1].speakers == 2)
            #expect(back.queue.items.allSatisfy { $0.state == .waiting })
        }

        @Test("an inbox recording whose transcript was deleted is not offered again")
        func forgottenInboxRecordingStaysOut() throws {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            let folder = rig.dir.appendingPathComponent("inbox")
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let memo = folder.appendingPathComponent("memo.m4a")
            FileManager.default.createFile(atPath: memo.path, contents: Data(count: 64))

            // What the delete does for a job whose source is in the inbox.
            #expect(Inbox.holds(memo.path, in: folder))
            let present = Inbox.contents(in: folder)
            #expect(Inbox.waiting(among: present, known: [], dismissed: []).count == 1)
            model.remember(dismissed: memo.path)
            #expect(Inbox.waiting(among: present, known: [],
                                  dismissed: Dismissed.load(from: rig.defaults)).isEmpty)
        }

        @Test("a row the person dismissed from the inbox is remembered, and a running one is not removed")
        func dismissing() {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            let memo = rig.add("memo.m4a", to: model)
            model.queue.items[0].collection = Inbox.collection
            model.dismiss(model.queue.items[0])
            #expect(model.queue.isEmpty)
            #expect(Dismissed.load(from: rig.defaults) == [canonicalPath(memo.url.path)])

            let running = rig.add("run.m4a", to: model)
            model.queue.mark(running.id, .running)
            model.dismiss(model.queue.items[0])
            #expect(model.queue.items.count == 1)
        }
    }

    /// The Mac kept awake, and the one notification, driven through the real model.
    @MainActor
    struct WorkTests {

        // MARK: staying awake

        @Test("the Mac is kept awake for a run, and let go when it finishes")
        func awakeForARun() async {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            let a = rig.add("a.m4a", to: model)
            #expect(rig.power.held == 0)
            model.start(a)
            #expect(rig.power.held == 1)
            #expect(await eventually { !model.engine.isRunning })
            #expect(rig.power.held == 0)
            #expect(rig.power.begun == ["Scriba is recording or transcribing"])
        }

        @Test("a run that fails lets go too")
        func awakeReleasedOnFailure() async {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            let bad = rig.add("bad.m4a", to: model)
            model.start(bad)
            #expect(rig.power.held == 1)
            #expect(await eventually { !model.engine.isRunning })
            #expect(rig.power.held == 0)
        }

        @Test("a run that cannot even be launched does not leave it held")
        func awakeReleasedWhenLaunchFails() async {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            let a = rig.add("a.m4a", to: model)
            Engine.pythonPath = rig.dir.appendingPathComponent("not-there").path
            model.start(a)
            #expect(!model.engine.isRunning)
            #expect(rig.power.held == 0)
        }

        @Test("Stop lets go at once")
        func awakeReleasedOnStop() async {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            let slow = rig.add("slow.m4a", to: model)
            model.start(slow)
            #expect(await eventually { rig.asked() == ["run slow.m4a"] })
            #expect(rig.power.held == 1)
            model.stopEverything()
            #expect(rig.power.held == 0)
            #expect(await eventually { !model.engine.isRunning })
            #expect(rig.power.held == 0)
        }

        @Test("quitting lets go, with the run killed and the model told")
        func awakeReleasedOnQuit() async {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            let slow = rig.add("slow.m4a", to: model)
            model.start(slow)
            #expect(await eventually { rig.asked() == ["run slow.m4a"] })
            model.engine.terminateChild()
            model.letTheMacSleep()
            #expect(rig.power.held == 0)
            #expect(await eventually { !model.engine.isRunning })
            #expect(rig.power.held == 0)
        }

        @Test("a batch holds it from the first recording to the last, and nothing is left after")
        func awakeAcrossABatch() async {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            for name in ["a.m4a", "b.m4a", "c.m4a"] { rig.add(name, to: model) }
            model.startAll()
            var heldWhileRunning = true
            #expect(await eventually {
                if model.engine.isRunning, rig.power.held != 1 { heldWhileRunning = false }
                return !model.runningAll && !model.engine.isRunning
            })
            #expect(heldWhileRunning)
            #expect(rig.power.held == 0)
        }

        @Test("a recording holds it while the microphone is open, and lets go when it stops")
        func awakeWhileRecording() async throws {
            let rig = Rig(); defer { rig.cleanup() }
            let folder = InputChangeTests.Folder()
            let engine = FakeEngine(format: InputChangeTests.format(48_000))
            let recorder = Recorder(directory: folder.url, defaults: folder.defaults,
                                    makeEngine: { engine }, microphoneAllowed: { true })
            let model = rig.model(recorder: recorder)
            #expect(rig.power.held == 0)
            await model.startRecording()
            #expect(recorder.isRecording)
            #expect(rig.power.held == 1)
            await InputChangeTests.play(engine, 1.2)
            #expect(await model.stopRecording() != nil)
            #expect(rig.power.held == 0)
        }

        @Test("a recording and a run together are one hold, let go only when both are over")
        func awakeForBoth() async throws {
            let rig = Rig(); defer { rig.cleanup() }
            let folder = InputChangeTests.Folder()
            let engine = FakeEngine(format: InputChangeTests.format(48_000))
            let recorder = Recorder(directory: folder.url, defaults: folder.defaults,
                                    makeEngine: { engine }, microphoneAllowed: { true })
            let model = rig.model(recorder: recorder)
            let slow = rig.add("slow.m4a", to: model)
            model.start(slow)
            await model.startRecording()
            #expect(rig.power.held == 1)
            await InputChangeTests.play(engine, 1.2)
            await model.stopRecording()
            #expect(rig.power.held == 1)        // the run is still going
            model.stopEverything()
            #expect(rig.power.held == 0)
            #expect(rig.power.begun.count == 1)
        }

        // MARK: a stop that is still finishing

        @Test("quitting while a stop is finishing waits for it: the file is closed and queued first")
        func quitDuringAStop() async throws {
            let rig = Rig(); defer { rig.cleanup() }
            let folder = InputChangeTests.Folder()
            let engine = FakeEngine(format: InputChangeTests.format(48_000))
            let live = FakeLive()
            live.finishTakes = 0.4
            let recorder = Recorder(directory: folder.url, defaults: folder.defaults,
                                    makeEngine: { engine }, microphoneAllowed: { true },
                                    makeLive: { _, _ in live })
            recorder.liveTextEnabled = true
            let model = rig.model(recorder: recorder)
            await model.startRecording()
            #expect(recorder.isRecording)
            await InputChangeTests.play(engine, 1.5)

            // The Stop button, then Cmd-Q in the second the analyzer takes to finish.
            let pressed = Task { await model.stopRecording() }
            await idle(0.1)
            #expect(recorder.isRecording)
            let quit = await model.stopRecording()

            let url = try #require(quit)
            #expect(!recorder.isRecording)
            #expect(live.finished)
            // Closed at its real length, and in the queue once, as it would be after a Stop.
            let file = try AVAudioFile(forReading: url)
            #expect(file.length >= 70_000 && file.length <= 74_000)
            #expect(model.queue.items.map(\.path) == [canonicalPath(url.path)])
            #expect(await pressed.value == url)
            #expect(model.queue.items.count == 1)
        }

        // MARK: the notification

        @Test("nobody is asked for permission until a run starts")
        func askedAtTheFirstStart() async {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            let a = rig.add("a.m4a", to: model)
            rig.add("b.m4a", to: model)
            #expect(rig.notifier.asked == 0)
            model.start(a)
            #expect(rig.notifier.asked == 1)
            #expect(await eventually { !model.engine.isRunning })
            // A second run, and a batch: the question was asked.
            model.startAll()
            #expect(await eventually { !model.runningAll && !model.engine.isRunning })
            #expect(rig.notifier.asked == 1)
        }

        @Test("a batch ends with one notification that counts, and names nothing")
        func batchNotification() async {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            for name in ["ada-lake-trip.m4a", "bad-rafiq.m4a", "ada-rafiq.m4a", "token-ada.m4a"] {
                rig.add(name, to: model)
            }
            model.startAll()
            #expect(await eventually { !model.runningAll && !model.engine.isRunning })
            #expect(rig.notifier.posted == [BatchNotice.Text(title: "Scriba",
                                                             body: "2 transcribed, 2 did not go through")])
            let text = rig.notifier.posted.map { $0.title + " " + $0.body }.joined().lowercased()
            for word in ["ada", "rafiq", "lake", "m4a", "token", "bad"] {
                #expect(!text.contains(word))
            }
        }

        @Test("a batch is told even with the window in front")
        func batchInFront() async {
            let rig = Rig(); defer { rig.cleanup() }
            rig.frontmost = true
            let model = rig.model()
            rig.add("a.m4a", to: model)
            rig.add("b.m4a", to: model)
            model.startAll()
            #expect(await eventually { !model.runningAll && !model.engine.isRunning })
            #expect(rig.notifier.posted.map(\.body) == ["2 transcribed"])
        }

        @Test("one recording on its own is told when the app is not in front, and not when it is")
        func singleRunsFollowTheWindow() async {
            let rig = Rig(); defer { rig.cleanup() }
            let model = rig.model()
            let a = rig.add("a.m4a", to: model)
            let b = rig.add("b.m4a", to: model)

            rig.frontmost = true
            model.start(a)
            #expect(await eventually { !model.engine.isRunning })
            await idle(0.2)
            #expect(rig.notifier.posted.isEmpty)

            // The person switched to another application while it ran.
            rig.frontmost = false
            model.start(b)
            #expect(await eventually { !model.engine.isRunning })
            #expect(rig.notifier.posted.map(\.body) == ["1 transcribed"])
        }

        @Test("a single failure in the background says that, and not why")
        func singleFailure() async {
            let rig = Rig(); defer { rig.cleanup() }
            rig.frontmost = false
            let model = rig.model()
            let bad = rig.add("bad.m4a", to: model)
            model.start(bad)
            #expect(await eventually { !model.engine.isRunning })
            #expect(rig.notifier.posted.map(\.body) == ["1 did not go through"])
        }

        @Test("recordings asked for one after another are one series, with one notification")
        func putNextIsOneSeries() async {
            let rig = Rig(); defer { rig.cleanup() }
            rig.frontmost = false
            let model = rig.model()
            let a = rig.add("a.m4a", to: model)
            let b = rig.add("b.m4a", to: model)
            model.start(a)
            model.start(b)
            #expect(await eventually { rig.asked().count == 2 && !model.engine.isRunning })
            await idle(0.3)
            #expect(rig.notifier.posted.map(\.body) == ["2 transcribed"])
        }

        @Test("Stop ends the series without a notification, and the next series starts from nothing")
        func stopSaysNothing() async {
            let rig = Rig(); defer { rig.cleanup() }
            rig.frontmost = false
            let model = rig.model()
            for name in ["a.m4a", "slow.m4a", "c.m4a"] { rig.add(name, to: model) }
            model.startAll()
            #expect(await eventually { rig.asked().contains("run slow.m4a") })
            model.stopEverything()
            #expect(await eventually { !model.engine.isRunning })
            await idle(0.3)
            #expect(rig.notifier.posted.isEmpty)
        }

        @Test("a recording started the moment after Stop is not counted together with what went before it")
        func stopThenStartAtOnce() async {
            let rig = Rig(); defer { rig.cleanup() }
            rig.frontmost = false
            let model = rig.model()
            for name in ["a.m4a", "slow.m4a", "c.m4a"] { rig.add(name, to: model) }
            model.startAll()
            #expect(await eventually { rig.asked().contains("run slow.m4a") })
            model.stopEverything()
            // Before the stopped process has been reaped: its end is never reported,
            // so the series it belonged to has to have been closed by the Stop.
            model.queue.items.removeAll { $0.url.lastPathComponent == "a.m4a" }
            let c = model.queue.items.first { $0.url.lastPathComponent == "c.m4a" }!
            model.start(c)
            #expect(await eventually { !model.engine.isRunning })
            await idle(0.3)
            #expect(rig.notifier.posted.map(\.body) == ["1 transcribed"])
        }

        @Test("quitting mid-run says nothing")
        func quitSaysNothing() async {
            let rig = Rig(); defer { rig.cleanup() }
            rig.frontmost = false
            let model = rig.model()
            let slow = rig.add("slow.m4a", to: model)
            model.start(slow)
            #expect(await eventually { rig.asked() == ["run slow.m4a"] })
            model.engine.terminateChild()
            #expect(await eventually { !model.engine.isRunning })
            await idle(0.3)
            #expect(rig.notifier.posted.isEmpty)
        }

        // MARK: the microphone going away

        @Test("a recording that ends by itself is queued, the person is told at the top of the window, and the window is asked to leave its pane")
        func recordingLost() async throws {
            let rig = Rig(); defer { rig.cleanup() }
            let folder = InputChangeTests.Folder()
            let engine = FakeEngine(format: InputChangeTests.format(48_000))
            let recorder = Recorder(directory: folder.url, defaults: folder.defaults,
                                    makeEngine: { engine }, microphoneAllowed: { true })
            recorder.restartPause = 0.01
            let model = rig.model(recorder: recorder)
            var left: [UUID?] = []
            let watch = model.recordingEnded.sink { left.append($0) }
            defer { watch.cancel() }

            await model.startRecording()
            await InputChangeTests.play(engine, 1.5)
            engine.startFailures = 99
            engine.changeInput(to: InputChangeTests.format(16_000), notify: false)
            await recorder.inputChanged()

            #expect(!recorder.isRecording)
            #expect(model.queue.items.count == 1)
            let item = try #require(model.queue.items.first)
            #expect(item.state == .waiting)
            #expect(item.note?.isWarning == true)
            #expect(item.note?.text.contains("could not be started again") == true)
            #expect(model.notice?.contains("could not be started again") == true)
            #expect(left == [item.id])
            #expect(rig.power.held == 0)
        }
    }

    /// The job list on screen against reads that were already in flight.
    @MainActor
    struct JobsStoreTests {

        @Test("an archive made while a read was in flight is still there when the read lands")
        func archiveSurvivesAnOlderRead() async {
            let rig = Rig(); defer { rig.cleanup() }
            rig.listAJob()
            let store = JobsStore()
            await store.reloadAndWait()
            #expect(store.jobs.map(\.jobDir) == ["/jobs/a"])
            #expect(store.jobs[0].archived == false)

            store.reload()                      // takes its snapshot: not archived
            await idle(0.15)
            await store.archive(jobDir: "/jobs/a", value: true)
            #expect(store.jobs[0].archived)

            // The old read lands during this, and must not undo the archive.
            var undone = false
            #expect(await eventually {
                if let job = store.jobs.first, !job.archived { undone = true }
                return !store.isLoading
            })
            #expect(!undone)
            #expect(store.jobs.first?.archived == true)
        }

        @Test("a deleted row stays deleted when a read that began before it lands")
        func deleteSurvivesAnOlderRead() async {
            let rig = Rig(); defer { rig.cleanup() }
            rig.listAJob()
            let store = JobsStore()
            await store.reloadAndWait()
            #expect(store.jobs.count == 1)

            store.reload()
            await idle(0.15)
            await store.forget(jobDir: "/jobs/a", withSource: false)
            #expect(store.jobs.isEmpty)

            var cameBack = false
            #expect(await eventually {
                if !store.jobs.isEmpty { cameBack = true }
                return !store.isLoading
            })
            #expect(!cameBack)
            #expect(store.jobs.isEmpty)
        }

        @Test("with nothing done in between, a read replaces the list as before")
        func plainReadStillLands() async {
            let rig = Rig(); defer { rig.cleanup() }
            let store = JobsStore()
            await store.reloadAndWait()
            #expect(store.jobs.isEmpty)
            rig.listAJob()
            await store.reloadAndWait()
            #expect(store.jobs.count == 1)
            #expect(store.problem == nil)
        }
    }
}
