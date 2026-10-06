import AppKit
import AVFoundation
import Combine
import Foundation

/// What has to outlive the window.
///
/// The engine, the queue, the job list and the recorder used to belong to the
/// window, and the README says the window can be closed while the work carries
/// on. It did carry on, in a process nobody could see: reopening built a new
/// queue that restored the running recording as waiting, with Transcribe live on
/// it (a second run on the same folder), the new window's engine became the one
/// quitting stopped, and when the first run ended it wrote its result into a
/// window that was gone. There is now one of these for the life of the app, made
/// by the AppDelegate and handed to each window. What stays in the window is
/// what is about the window: the selection, the filter, the dialogs.
///
/// The model does not know what is selected. Where something it does needs the
/// window to react, it says so on a subject and the window decides.
@MainActor
final class AppModel: ObservableObject {
    let engine = Engine()
    let store = JobsStore()
    let recorder: Recorder

    @Published var queue = Queue() {
        didSet { saveIfChanged() }
    }

    /// "Transcribe all" is on: when one recording ends, the next waiting one
    /// starts. Not set by Transcribe on a single recording, nor by asking for one
    /// to go after the current one: those do what they say and stop.
    @Published private(set) var runningAll = false

    /// The recordings somebody asked to run after the current one, in the order
    /// they asked. They run one after the other when it ends, and then the queue
    /// stops. This used to be `runningAll`, so putting one recording next started
    /// every recording that was waiting, which is the thing this app promises
    /// never to do.
    @Published private(set) var upNext: [UUID] = []

    /// Recordings were asked for after the current one and the run of them is not
    /// over. Stays set while the last of them runs too, which `upNext` does not:
    /// a recording leaves it as it starts. The failure alert is held back for as
    /// long as this is set, exactly as it is for Transcribe all, and one summary
    /// is made when the run of them ends.
    @Published private(set) var chaining = false

    /// The language for the microphone. A recording made here carries it into
    /// the queue; a dropped file starts on automatic and has its own.
    @Published var language = "auto"

    @Published private(set) var scanning: String?

    /// Something the person has to be told that belongs to no recording: the
    /// microphone would not start, or what was recorded was too short to keep.
    /// Shown at the top of the window until it is closed. Nothing on screen
    /// changed when Record failed, so the button looked dead.
    @Published var notice: String?

    /// Where the queue is written between launches. It lived in memory only, so
    /// quitting with ten recordings waiting meant finding and dropping the ten
    /// again, and a crash mid-batch lost the list without a word.
    static let savedQueueKey = "queuedRecordings"

    /// Inbox recordings somebody swiped out of the queue. Kept across launches,
    /// or the next look at the inbox would offer them straight back.
    private var dismissed: Set<String>

    /// The failures of the batch or the chain that is running, for the one message
    /// at its end.
    private var batchFailures: [(name: String, reason: String)] = []

    private let defaults: UserDefaults
    private var lastSaved: [Queue.Saved] = []

    // MARK: - staying awake, and saying when it is over

    /// The one place the Mac is kept awake from. It follows the engine and the
    /// recorder (see `init`) and nothing else asks it anything, except the quit,
    /// which lets go by hand because the app is about to stop watching.
    private let awake: StayAwake

    private let notifier: BatchNotifying
    private let isFrontmost: () -> Bool
    private var asked = false

    /// Recordings that ended, one outcome each, since the series of runs that is
    /// going on now began. A series is what one press of Transcribe, or one press
    /// of Transcribe all, sets going, including the recordings asked for after
    /// the current one: it ends when nothing follows a run, and says so once.
    private var outcomes: [Engine.Outcome] = []
    private var inSeries = false
    /// Transcribe all was pressed during the series. That is the person asking
    /// to be told when the whole list is done, whether or not they are looking.
    private var wasBatch = false

    private var watching: Set<AnyCancellable> = []

    // MARK: - telling the window

    /// A finished recording has left the queue. `jobDir` is the job it produced,
    /// when the list has it. The window moves the selection only if the person
    /// was watching that recording, which only the window can tell.
    struct Followed {
        let itemID: UUID
        let jobDir: String?
    }
    let followed = PassthroughSubject<Followed, Never>()

    /// Recordings were added by hand or by recording. The window selects the
    /// first one if nothing is selected.
    let selectFirst = PassthroughSubject<Void, Never>()

    /// A recording ended without anybody pressing Stop: the microphone went away
    /// and could not be started again. The id is the row the kept recording
    /// became, or nil when there was too little to keep. The window leaves the
    /// recording pane, which has nothing left to show.
    let recordingEnded = PassthroughSubject<UUID?, Never>()

    /// The last four are the real ones unless a test puts something in their
    /// place. They are made here and not as default arguments, which cannot make
    /// a main-actor object.
    init(defaults: UserDefaults = .standard,
         awake: StayAwake? = nil,
         notifier: BatchNotifying? = nil,
         isFrontmost: (() -> Bool)? = nil,
         recorder: Recorder? = nil) {
        self.defaults = defaults
        self.recorder = recorder ?? Recorder()
        self.awake = awake ?? StayAwake()
        self.notifier = notifier ?? SystemNotifier()
        self.isFrontmost = isFrontmost ?? { NSApplication.shared.isActive }
        self.dismissed = Dismissed.load(from: defaults)
        restoreQueue()

        // Awake while there is work, however the work ends. Both publishers say
        // the new value as they change, so this reads what they emit and not the
        // property they are about to set. Following the state rather than the
        // calls that start and stop is what lets a run that fails, is stopped, or
        // is killed by the quit let go without anyone remembering to.
        Publishers.CombineLatest(engine.$isRunning, self.recorder.$state)
            .map { running, state in running || state == .recording }
            .removeDuplicates()
            .sink { [weak self] busy in self?.awake.set(busy) }
            .store(in: &watching)

        self.recorder.stoppedItself
            .sink { [weak self] url in self?.recorderStoppedItself(url) }
            .store(in: &watching)
    }

    /// The app is leaving. The process ending would drop the assertion anyway;
    /// this says so first.
    func letTheMacSleep() { awake.release() }

    // MARK: - looking around

    /// Re-read the list, then look in the inbox for recordings nobody has
    /// dealt with and put them in the queue, as waiting. Nothing starts.
    func lookAround() {
        Task {
            await store.reloadAndWait()
            adoptInbox()
        }
    }

    private func adoptInbox() {
        let known = Set(store.jobs.map { canonicalPath($0.sourcePath) })
        // What takes a waiting row out is a job that finished, or one another
        // run is on right now. A stopped run leaves a folder too, and counting
        // it made Stop throw the recording out of the queue at the next look
        // at the inbox, into "Not finished".
        let settled = Set(store.jobs.filter { $0.isFinished || $0.isRunningElsewhere }
                                    .map { canonicalPath($0.sourcePath) })
        let folder = Inbox.folder
        Task.detached(priority: .utility) {
            let present = Inbox.contents(in: folder)
            await MainActor.run {
                // A file gone from the inbox has nothing left to be dismissed
                // from; forgetting it keeps the list short and lets a recording
                // copied in a second time be offered a second time.
                Dismissed.prune(keeping: Set(present.map { canonicalPath($0.url.path) }),
                                in: self.defaults)
                self.dismissed = Dismissed.load(from: self.defaults)
                // Rows the engine reached by another road are no longer waiting
                // on this window: the job list has them now, running or done.
                self.queue.reconcileInbox(known: settled)
                self.enqueue(Inbox.waiting(among: present, known: known,
                                           dismissed: self.dismissed),
                             select: false)
            }
        }
    }

    /// Remember that somebody does not want this inbox recording offered again.
    func remember(dismissed path: String) {
        Dismissed.add(path, to: defaults)
        dismissed.insert(canonicalPath(path))
    }

    /// Take a row out of the queue. An inbox recording is also remembered as
    /// dismissed, or it would be back on the next look at the inbox.
    func dismiss(_ item: QueueItem) {
        guard item.state != .running else { return }
        queue.remove(item.id)
        upNext.removeAll { $0 == item.id }
        if item.collection == Inbox.collection { remember(dismissed: item.path) }
    }

    // MARK: - surviving a quit

    /// Written by the model, not by a window: a run that ends with the window
    /// closed changes the queue too, and a list saved only while a window was
    /// open went on saying "waiting" over a finished recording.
    private func saveIfChanged() {
        let saved = queue.saved
        guard saved != lastSaved else { return }
        lastSaved = saved
        if let data = try? JSONEncoder().encode(saved) {
            defaults.set(data, forKey: Self.savedQueueKey)
        }
    }

    private func restoreQueue() {
        guard queue.isEmpty,
              let data = defaults.data(forKey: Self.savedQueueKey),
              let saved = try? JSONDecoder().decode([Queue.Saved].self, from: data)
        else { return }
        queue = Queue.restore(saved)
        for item in queue.items { measure(item.id, item.url) }
    }

    // MARK: - adding

    /// Read the length once, off the main actor, and remember it on the item.
    private func measure(_ id: UUID, _ url: URL) {
        Task.detached(priority: .utility) {
            let asset = AVURLAsset(url: url)
            guard let duration = try? await asset.load(.duration) else { return }
            let seconds = CMTimeGetSeconds(duration)
            guard seconds.isFinite, seconds > 0 else { return }
            await MainActor.run {
                if let i = self.queue.index(of: id) { self.queue.items[i].minutes = seconds / 60 }
            }
        }
    }

    /// Add files, or everything under a folder, to the queue.
    ///
    /// Folders are walked off the main thread. A tree of a few hundred recordings
    /// means a few hundred trips to the filesystem, and doing that while the
    /// window is drawing is how a drag turns into a beachball.
    func add(_ urls: [URL]) {
        var files: [(url: URL, collection: String)] = []
        var folders: [URL] = []
        for url in urls {
            var isDirectory: ObjCBool = false
            let there = FileManager.default.fileExists(atPath: url.path,
                                                       isDirectory: &isDirectory)
            if there && isDirectory.boolValue {
                folders.append(url)
            } else if Library.isRecording(url) {
                files.append((url, ""))
            }
        }
        enqueue(files)

        guard !folders.isEmpty else { return }
        scanning = folders.count == 1
            ? "Looking through \(folders[0].lastPathComponent)…"
            : "Looking through \(folders.count) folders…"
        Task.detached(priority: .userInitiated) {
            let found = folders.flatMap { Library.recordings(under: $0) }
            await MainActor.run {
                self.scanning = nil
                self.enqueue(found)
            }
        }
    }

    /// The queued recordings that are new, in the order they were found.
    func enqueue(_ found: [(url: URL, collection: String)],
                 language: String = "auto", select: Bool = true) {
        let added = queue.add(found, language: language)
        guard !added.isEmpty else { return }
        // Read the durations afterwards and one at a time. Doing it inside the
        // add opened every container before the list had drawn once.
        for item in added { measure(item.id, item.url) }
        if select { selectFirst.send() }
    }

    /// The same language and head count for every recording still waiting.
    /// The settings belong to one recording each; this is the shortcut for a
    /// batch that is all one meeting.
    func applyToAllWaiting(language: String, speakers: Int) {
        for i in queue.items.indices where queue.items[i].state == .waiting {
            queue.items[i].language = language
            queue.items[i].speakers = speakers
        }
    }

    // MARK: - recording

    /// Start the microphone, and say so if it would not start. Permission denied
    /// and no input device both end with the recorder in a failed state that
    /// nothing read.
    func startRecording() async {
        notice = nil
        await recorder.start(language: language)
        if case .failed(let message) = recorder.state { notice = message }
    }

    /// Stop the microphone and hand what it recorded to the queue. The Stop
    /// button and quitting both end here.
    ///
    /// Stopping does not start the transcription. A recording that has just ended
    /// is the moment somebody is most likely to want to say how many people were
    /// in the room, or to record the next one straight away; a run that began on
    /// its own would hold the machine for as long as the conversation lasted.
    @discardableResult
    func stopRecording() async -> URL? {
        let wasRecording = recorder.isRecording
        guard let url = await recorder.stop() else {
            // Nothing was kept, and the recorder's reason was drawn only in the
            // recording pane, which is gone by now.
            if wasRecording { notice = recorder.warning }
            return nil
        }
        keep(url)
        return url
    }

    /// Put a finished recording in the queue, with its warning if it has one.
    @discardableResult
    private func keep(_ url: URL) -> UUID? {
        // The language chosen for the microphone is the language of what was just
        // recorded.
        enqueue([(url: url, collection: "")], language: language)
        // A silent recording is kept, and the warning goes with its row: that is
        // where the person lands, and it stays true after the next recording.
        guard let i = queue.items.firstIndex(where: { $0.path == canonicalPath(url.path) })
        else { return nil }
        if let warning = recorder.warning {
            queue.items[i].note = QueueItem.Note(text: warning, isWarning: true)
        }
        return queue.items[i].id
    }

    /// The recorder closed the file by itself, because the microphone it was
    /// using went away and it could not be restarted on another. What was
    /// recorded up to then is queued exactly as a Stop would have queued it.
    private func recorderStoppedItself(_ url: URL?) {
        let id = url.flatMap { keep($0) }
        // At the top of the window, whatever is selected: the person who plugged
        // something in is not necessarily looking at the recording pane, and a
        // recording that ended on its own is something they have to be told.
        notice = recorder.warning
        recordingEnded.send(id)
    }

    // MARK: - running

    /// The recordings asked for after the current one, still waiting, in order.
    var askedNext: [QueueItem] {
        upNext.compactMap { queue.item($0) }.filter { $0.state == .waiting }
    }

    /// More runs follow by themselves: Transcribe all, or recordings asked for
    /// after the current one.
    private var runsFollow: Bool { runningAll || chaining }

    /// What the failure alert says, or nil. Held back for as long as a batch
    /// runs: every failing file raised the alert and the next file's start took
    /// it down again, so the batch flashed alerts nobody could read and left the
    /// last one. The reasons are on the rows meanwhile, and the batch says it
    /// once when it ends. Recordings asked for after the current one are the same
    /// thing in a smaller size: a failure followed by one of them lost its alert
    /// the moment the next run started.
    var alertText: String? { runsFollow ? nil : engine.errorText }

    /// Returns false when nothing was started or queued, so a caller that had
    /// armed the queue first can put it back.
    @discardableResult
    func start(_ item: QueueItem) -> Bool {
        guard Queue.canStart(item.state, engineBusy: false), !engine.isNaming else { return false }
        if engine.isRunning {
            // Asked while another one runs: this one goes next and then it stops.
            // Pressing the button used to do nothing, silently, which is the worst
            // thing a button can do. The others stay where they are: only
            // Transcribe all (or Cmd-T) carries on through the list.
            queue.mark(item.id, .waiting)
            if !upNext.contains(item.id) { upNext.append(item.id) }
            chaining = true
            // Front of the line in the order they were asked, so the sidebar
            // draws them in the order they will run.
            for id in upNext.reversed() { queue.moveToFront(id) }
            return true
        }
        upNext.removeAll { $0 == item.id }
        beginSeries()
        mark(item.id, .running)
        engine.run(file: item.url, language: item.language,
                   minSpeakers: item.speakers > 0 ? item.speakers : nil,
                   maxSpeakers: item.speakers > 0 ? item.speakers : nil,
                   collection: item.collection) { [self] outcome in
            switch outcome {
            case .finished:
                mark(item.id, .finished)
                // Follow it into the list. The pane used to fall back to the
                // setup screen, offering to transcribe again a recording that
                // had just been transcribed, while the row sat under "Waiting".
                followFinished(item)
            case .failed:
                // The reason, not the fact. "Did not finish" was true and useless,
                // and the alert carrying the real message is dismissed once.
                let reason = engine.failureSummary
                mark(item.id, .failed(reason))
                if runsFollow {
                    batchFailures.append((name: item.url.lastPathComponent, reason: reason))
                }
                // The engine said nothing the app could read, so the row carries
                // the tail of a log. It wrote the real reason into the job.
                if engine.failureReason == nil { readReasonFromJob(item, replacing: reason) }
            case .cancelled:
                // Asked for. Back in the queue rather than marked as broken.
                mark(item.id, .waiting)
            }
            if case .finished = outcome {} else { store.reload() }
            if outcome != .cancelled {
                outcomes.append(outcome)
                carryOn()
            }
            settleSeries(after: outcome)
        }
        return true
    }

    /// The first run started while nothing was running opens a series; the runs
    /// that follow it by themselves belong to it.
    private func beginSeries() {
        guard !inSeries else { return }
        inSeries = true
        outcomes = []
        wasBatch = runningAll
        // Not at launch: somebody who only looks at transcripts is never asked.
        // The system remembers the answer, so this asks once ever, and the flag
        // keeps it to once for each launch while the question is still open.
        if !asked {
            asked = true
            notifier.askIfUndecided()
        }
    }

    /// A run ended and nothing followed it: the series is over. Say how it went,
    /// once, to somebody who is not looking.
    ///
    /// Transcribe all always says it: the person asked to be told when the list
    /// is done. A single recording says it only when the app is not in front. In
    /// front, the system keeps a notification from an application that is
    /// being used from showing anyway, and the result is on screen.
    private func settleSeries(after outcome: Engine.Outcome) {
        guard inSeries, !engine.isRunning else { return }
        let text = BatchNotice.text(for: outcomes)
        let tell = outcome != .cancelled && (wasBatch || !isFrontmost())
        closeSeries()
        if tell, let text { notifier.post(text) }
    }

    private func closeSeries() {
        inSeries = false
        outcomes = []
        wasBatch = false
    }

    func startAll() {
        // A save of names is running and nothing can start behind it. Arming the
        // queue anyway left it armed: the next recording somebody started on its
        // own then carried on through the whole list when it ended.
        guard !engine.isNaming else { return }
        if engine.isRunning {
            // Mid-run it means: let the waiting ones follow. What the recordings
            // asked for after this one have already failed stays in the tally.
            if !runningAll && !queue.waiting.isEmpty {
                runningAll = true
                wasBatch = true
            }
            return
        }
        batchFailures = []
        runningAll = queue.waiting.count > 1
        next()
    }

    /// What follows a run that ended by itself: the rest of the batch, if
    /// Transcribe all was pressed, else the recordings that were asked for.
    private func carryOn() {
        if runningAll {
            next()
            return
        }
        upNext.removeAll { queue.item($0)?.state != .waiting }
        guard let id = upNext.first, let item = queue.item(id), start(item) else {
            endChain()
            return
        }
    }

    /// The recordings that were asked for have all run, or none could be started.
    private func endChain() {
        guard chaining else { return }
        chaining = false
        summariseFailures()
    }

    /// One at a time. Whisper already spreads across every fast CPU, so two
    /// transcriptions at once finish later than the same two in sequence, and the
    /// machine is unusable while they run.
    private func next() {
        guard let item = queue.nextToRun else {
            endBatch()
            return
        }
        // Refused: nothing runs, so nothing will end to carry the batch on, and
        // the queue must not stay armed for the next single recording.
        if !start(item) { runningAll = false }
    }

    private func endBatch() {
        guard runningAll else { return }
        runningAll = false
        chaining = false
        summariseFailures()
    }

    /// Say once what did not go through, and forget it.
    private func summariseFailures() {
        if let summary = Queue.batchSummary(batchFailures) { engine.errorText = summary }
        batchFailures = []
    }

    /// Stop the running job and empty the rest of the queue.
    ///
    /// Stopping one and starting the next is not what anybody means by Stop, and
    /// the queue was started by one press of one button, so it ends the same way.
    /// Everything still waiting goes back to waiting rather than being thrown
    /// away: the files are still listed and one more press starts them again.
    func stopEverything() {
        runningAll = false
        upNext = []
        chaining = false
        batchFailures = []
        // Somebody pressed Stop and is looking at the window: no notification.
        closeSeries()
        engine.cancel()
        queue.stopAll()
    }

    /// The last resort for a failed row: the reason the engine wrote into the
    /// job's state.json (`failed`), read through the list, which is how the app
    /// reads every job. The list hides a job whose recording is queued, but not
    /// from the model. A read that was already running may predate the failure,
    /// so a second one is made before giving up.
    private func readReasonFromJob(_ item: QueueItem, replacing old: String) {
        Task {
            var written = ""
            for _ in 0..<2 where written.isEmpty {
                await store.reloadAndWait()
                written = store.jobs.first { canonicalPath($0.sourcePath) == item.path }?.failed ?? ""
            }
            guard !written.isEmpty else { return }
            let better = Engine.summarise(Engine.oneLine(written))
            guard better != old else { return }
            // Only a row still saying what it said, and a batch still counting.
            if case .failed(let now)? = queue.item(item.id)?.state, now == old {
                queue.mark(item.id, .failed(better))
            }
            let name = item.url.lastPathComponent
            if let i = batchFailures.firstIndex(where: { $0.name == name && $0.reason == old }) {
                batchFailures[i].reason = better
            }
        }
    }

    private func mark(_ id: UUID, _ state: QueueItem.State) {
        queue.mark(id, state)
    }

    /// Take the finished recording out of the queue, once the list has been
    /// re-read to include the job it produced.
    private func followFinished(_ item: QueueItem) {
        Task {
            await store.reloadAndWait()
            let path = item.path
            // The row leaves whether or not the job is found: a finished
            // recording has nothing left to wait for. Matching by the engine's
            // spelling of the path, not the Finder's, is what finds it.
            let job = store.jobs.first { canonicalPath($0.sourcePath) == path }
            queue.remove(item.id)
            followed.send(Followed(itemID: item.id, jobDir: job?.jobDir))
        }
    }
}
