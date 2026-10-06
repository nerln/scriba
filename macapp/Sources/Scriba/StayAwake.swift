import Foundation

/// Keeps the Mac from falling asleep on its own while Scriba is recording or
/// transcribing.
///
/// A transcription takes about as long as the recording lasted and a batch takes
/// the night. A Mac nobody touches for a few minutes goes to idle sleep, the
/// engine's process is frozen with the model halfway through the file, and
/// nothing on screen says why it is late. A conversation recorded on a laptop
/// nobody is touching ends the same way, at the first idle sleep, with half of
/// it on disk.
///
/// There is one of these, owned by the AppModel, and it is told only whether
/// the app is busy. The model decides that from the engine and the recorder
/// themselves (see `AppModel.init`) instead of from the places that start and
/// stop them, so a run that ends by finishing, failing, being stopped or being
/// killed by the quit lets go through the same door as every other.
///
/// Only idle sleep is refused. The Sleep menu still works, and so does closing
/// the lid: a laptop sleeps when its lid is closed whatever an application asked
/// for (unless it is on power with an external display), and this does not try to
/// change that. The screen is also still allowed to go dark.
@MainActor
final class StayAwake {
    /// What the system is asked, kept as two closures so a test can count the
    /// holds and the releases without touching the real power assertions.
    typealias Begin = (_ reason: String) -> NSObjectProtocol
    typealias End = (NSObjectProtocol) -> Void

    /// Shown by `pmset -g assertions`. Nothing from a recording goes in it: it is
    /// readable by every program on the Mac.
    static let reason = "Scriba is recording or transcribing"

    private let begin: Begin
    private let end: End
    private var token: NSObjectProtocol?

    init(begin: @escaping Begin = { reason in
             ProcessInfo.processInfo.beginActivity(
                 options: [.idleSystemSleepDisabled, .userInitiated], reason: reason)
         },
         end: @escaping End = { ProcessInfo.processInfo.endActivity($0) }) {
        self.begin = begin
        self.end = end
    }

    var isHeld: Bool { token != nil }

    /// Hold while `wanted`, let go when not. Saying the same thing twice does
    /// nothing, so a caller never has to remember what it said last.
    func set(_ wanted: Bool) {
        if wanted { hold() } else { release() }
    }

    func hold() {
        guard token == nil else { return }
        token = begin(Self.reason)
    }

    func release() {
        guard let held = token else { return }
        token = nil
        end(held)
    }
}
