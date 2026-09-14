import Foundation

/// The folder other tools drop recordings into, and what is new in it.
///
/// `scriba memos` and the paranco mirror both put files in `~/.scriba/inbox`
/// (watch.py, INBOX) and then wait for somebody to transcribe them. Until now
/// that somebody had to be a terminal: the app did not know the folder existed,
/// so a recording that came off the phone was invisible in the window until a
/// job for it appeared on disk. The app now lists what is there as waiting,
/// which is what it is. Nothing starts on its own; that part has not changed.
enum Inbox {
    /// Named in the sidebar and written into the job, so a recording that
    /// arrived this way stays recognisable after it has been transcribed.
    static let collection = "Inbox"

    /// The same folder the engine uses, honouring SCRIBA_HOME the same way.
    static var folder: URL {
        let env = ProcessInfo.processInfo.environment
        let home = env["SCRIBA_HOME"].map { URL(fileURLWithPath: $0) }
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(".scriba")
        return home.appendingPathComponent("inbox")
    }

    /// Every recording in the folder, whoever has dealt with it.
    static func contents(in folder: URL = folder) -> [(url: URL, collection: String)] {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: folder.path, isDirectory: &isDir),
              isDir.boolValue else { return [] }
        return Library.recordings(under: folder)
            .map { (url: $0.url, collection: collection) }
    }

    /// The recordings nobody has dealt with: no job on disk for them, and not
    /// taken out of the list by hand.
    ///
    /// `known` is the set of source paths the job list reports, spelled by
    /// `canonicalPath`. `dismissed` is what was swiped away: a file somebody
    /// removed from the queue must not come back on the next glance, or the
    /// swipe did nothing.
    static func waiting(among present: [(url: URL, collection: String)],
                        known: Set<String>,
                        dismissed: Set<String>) -> [(url: URL, collection: String)] {
        present.filter { entry in
            let path = canonicalPath(entry.url.path)
            return !known.contains(path) && !dismissed.contains(path)
        }
    }

    static func waiting(in folder: URL = folder,
                        known: Set<String>,
                        dismissed: Set<String>) -> [(url: URL, collection: String)] {
        waiting(among: contents(in: folder), known: known, dismissed: dismissed)
    }
}

/// The files somebody took out of the queue by hand, kept across launches.
///
/// Only inbox files need remembering: a dropped file that is removed is simply
/// not dropped again, but the inbox is looked at on every return to the window
/// and would offer the same file back every time.
enum Dismissed {
    static let key = "dismissedInboxRecordings"

    static func load(from defaults: UserDefaults = .standard) -> Set<String> {
        Set(defaults.stringArray(forKey: key) ?? [])
    }

    static func add(_ path: String, to defaults: UserDefaults = .standard) {
        var all = load(from: defaults)
        all.insert(canonicalPath(path))
        defaults.set(Array(all).sorted(), forKey: key)
    }

    /// A file that is gone from the inbox has nothing left to be dismissed
    /// from. Forgetting it keeps the list from growing for ever, and lets a
    /// recording that is copied in again a second time be offered again.
    static func prune(keeping present: Set<String>, in defaults: UserDefaults = .standard) {
        let kept = load(from: defaults).intersection(present)
        defaults.set(Array(kept).sorted(), forKey: key)
    }
}
