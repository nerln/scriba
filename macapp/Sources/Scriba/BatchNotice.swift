import Foundation
import UserNotifications

/// The one notification when a run of recordings ends: how many went through and
/// how many did not, and nothing else.
///
/// Numbers only, on purpose. A notification is drawn on the lock screen and in
/// the Notification Center, where anybody standing behind the Mac reads it. A
/// file name is a person's name or the subject of a meeting often enough that
/// none is ever put here, and neither is a word from a recording or the reason
/// something failed (that is in the window, where it was already).
enum BatchNotice {
    struct Text: Equatable {
        let title: String
        let body: String
    }

    /// What to say about the recordings of one run, or nil when there is
    /// nothing to say. A recording that was stopped by hand is not an outcome:
    /// somebody pressed Stop and knows.
    static func text(for outcomes: [Engine.Outcome]) -> Text? {
        let transcribed = outcomes.filter { $0 == .finished }.count
        let failed = outcomes.filter { $0 == .failed }.count
        var parts: [String] = []
        if transcribed > 0 { parts.append("\(transcribed) transcribed") }
        if failed > 0 { parts.append("\(failed) did not go through") }
        guard !parts.isEmpty else { return nil }
        return Text(title: "Scriba", body: parts.joined(separator: ", "))
    }
}

/// What the model needs from the notification centre, so the model can be run
/// with no centre at all.
@MainActor
protocol BatchNotifying: AnyObject {
    /// Ask the system for permission to show notifications, if nobody has asked
    /// yet. The model calls it when work starts, not at launch: the person has
    /// just pressed a button that runs for a long time, which is the moment a
    /// request to be told when it is over makes sense.
    func askIfUndecided()
    /// Show it, if the person allowed notifications. Refused or never answered
    /// means nothing happens, and nothing else in the app changes.
    func post(_ text: BatchNotice.Text)
}

@MainActor
final class SystemNotifier: BatchNotifying {
    /// The same identifier every time, so a notification still on the screen from
    /// the last run is replaced by this one instead of piling up.
    private static let identifier = "dev.nerelli.scriba.batch-end"

    /// Nil for anything that is not an application bundle. Asking
    /// `UNUserNotificationCenter` for its current centre from a bare executable
    /// (`swift run`, the test runner) is an exception, not an error, and it ends
    /// the process.
    private var center: UNUserNotificationCenter? {
        guard Bundle.main.bundleURL.pathExtension == "app" else { return nil }
        return UNUserNotificationCenter.current()
    }

    func askIfUndecided() {
        guard let center else { return }
        Task {
            let settings = await center.notificationSettings()
            guard settings.authorizationStatus == .notDetermined else { return }
            _ = try? await center.requestAuthorization(options: [.alert, .sound])
        }
    }

    func post(_ text: BatchNotice.Text) {
        guard let center else { return }
        Task {
            let allowed = await center.notificationSettings().authorizationStatus
            guard allowed == .authorized || allowed == .provisional else { return }
            let content = UNMutableNotificationContent()
            content.title = text.title
            content.body = text.body
            content.sound = .default
            // No category and so no buttons: pressing it only brings the app forward.
            let request = UNNotificationRequest(identifier: Self.identifier,
                                                content: content, trigger: nil)
            try? await center.add(request)
        }
    }
}
