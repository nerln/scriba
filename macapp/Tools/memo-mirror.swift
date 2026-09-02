// Copies new voice memos out of Apple's library and does nothing else.
//
// Why this exists as a separate program. Reading the Voice Memos library needs
// Full Disk Access, and the alternative was granting it to whatever runs scriba.
// That means a Python interpreter, and Full Disk Access on a Python interpreter
// is Full Disk Access for every package ever installed into that environment.
// This is sixty lines that can be read in a minute, it copies audio one way, and
// it is the only thing that ever needs the permission.
//
// It also cannot be used as a back door by the rest of scriba, because macOS
// attributes an access to the process that asked for it. A terminal driving this
// through `open` or AppleScript gets the terminal's permissions, not this one's,
// which is why the launch agent starts it rather than scriba starting it.
//
//   swiftc -O -parse-as-library memo-mirror.swift -o memo-mirror
//   memo-mirror --check      says whether it can see the library
//   memo-mirror              copies what is new and exits
//   memo-mirror --watch 60   the same, every sixty seconds

import Foundation

let library = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent("Library/Group Containers/group.com.apple.VoiceMemos.shared/Recordings")
let inbox = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent(".scriba/inbox")

let audio: Set<String> = ["m4a", "mp3", "wav", "aiff", "aifc", "caf"]

/// A refusal that carries the sentence a person needs, rather than an error code.
struct Refused: Error { let message: String }

/// Whether the library can be listed, and what it holds.
func recordings() -> Result<[URL], Refused> {
    do {
        let all = try FileManager.default.contentsOfDirectory(
            at: library, includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles])
        return .success(all.filter { audio.contains($0.pathExtension.lowercased()) })
    } catch let error as NSError where error.code == NSFileReadNoPermissionError {
        return .failure(Refused(message: """
            macOS is not letting this copy read the Voice Memos library.
            Give Full Disk Access to ScribaMemoMirror.app in System Settings >
            Privacy & Security > Full Disk Access, then run it again.
            Nothing else in scriba needs that permission.
            """))
    } catch {
        return .failure(Refused(message: "cannot read the library: \(error.localizedDescription)"))
    }
}

/// Copy what is not already there. Same name and same size counts as there.
func mirror() -> Int {
    guard case .success(let found) = recordings() else { return 0 }
    try? FileManager.default.createDirectory(at: inbox, withIntermediateDirectories: true)

    var copied = 0
    for source in found {
        let destination = inbox.appendingPathComponent(source.lastPathComponent)
        let theirs = (try? source.resourceValues(forKeys: [.fileSizeKey]))?.fileSize
        let mine = (try? destination.resourceValues(forKeys: [.fileSizeKey]))?.fileSize
        if mine != nil && mine == theirs { continue }
        // A recording still being written by the app, or still coming down from
        // iCloud, is left for the next pass rather than copied half finished.
        if mine != nil { try? FileManager.default.removeItem(at: destination) }
        do {
            try FileManager.default.copyItem(at: source, to: destination)
            copied += 1
            print("copied \(source.lastPathComponent)")
        } catch {
            print("could not copy \(source.lastPathComponent): \(error.localizedDescription)")
        }
    }
    return copied
}

@main
struct Main {
    static func main() {
        let args = Array(CommandLine.arguments.dropFirst())

        switch recordings() {
        case .failure(let why):
            print(why.message)
            exit(1)
        case .success(let found):
            if args.contains("--check") {
                print("the library is readable: \(found.count) recordings")
                print("they are mirrored into \(inbox.path)")
                exit(0)
            }
        }

        if let i = args.firstIndex(of: "--watch"), i + 1 < args.count,
           let seconds = Double(args[i + 1]) {
            while true {
                _ = mirror()
                Thread.sleep(forTimeInterval: seconds)
            }
        }
        let n = mirror()
        print(n == 0 ? "nothing new" : "\(n) new recordings in \(inbox.path)")
    }
}
