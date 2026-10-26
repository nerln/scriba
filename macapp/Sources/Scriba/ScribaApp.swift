import SwiftUI

/// The entry point picks between the window and the one command-line check.
///
/// SwiftUI's App protocol supplies its own `main`, so intercepting the arguments
/// means owning the entry point and calling it. The check has to live in this
/// binary rather than a tool of its own: it drives LiveTranscriber, and a second
/// executable target cannot depend on an executable target.
@main
struct Entry {
    static func main() {
        let args = CommandLine.arguments
        if let i = args.firstIndex(of: "--live-check"), i + 1 < args.count {
            var language = "en"
            if let l = args.firstIndex(of: "--lang"), l + 1 < args.count {
                language = args[l + 1]
            }
            LiveCheck.run(path: args[i + 1], language: language)
        }
        ScribaApp.main()
    }
}

struct ScribaApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @Environment(\.openWindow) private var openWindow

    var body: some Scene {
        // The delegate hears the Dock icon being clicked and has to open the window
        // then, and only the App can read the action that does it. Handed over on
        // every evaluation, so it is there before anything can ask.
        let _ = { delegate.showWindow = { openWindow(id: "main") } }()

        // One window, not a group of them. A WindowGroup made a new window on
        // demand and had no way to bring a closed one back: once it was closed,
        // nothing in the menus opened another (the command for it is removed
        // below) and neither did a click on the Dock icon, while the run it had
        // been showing carried on. A Window is the one window, which macOS can
        // reopen, and a second one cannot exist.
        Window("Scriba", id: "main") {
            ContentView(model: delegate.model)
        }
        .windowToolbarStyle(.unified)
        .commands {
            CommandGroup(replacing: .newItem) { }
            // Where somebody looks for a window that is not there. The system lists
            // a window that is open, not one that was closed.
            CommandGroup(after: .windowList) {
                Button("Show Scriba") { openWindow(id: "main") }
            }
            CommandGroup(after: .toolbar) {
                Button("Refresh the list") {
                    NotificationCenter.default.post(name: .scribaRefresh, object: nil)
                }
                .keyboardShortcut("r", modifiers: .command)
            }
            // The two things this app does, on the keys a Mac application puts
            // them on. Reaching for the mouse to start something that then runs
            // for an hour is the wrong shape.
            CommandMenu("Transcribe") {
                Button("Record a conversation") {
                    NotificationCenter.default.post(name: .scribaRecord, object: nil)
                }
                // Shift is not decoration: command-R is Refresh, and starting a
                // recording by accident while reaching for it would be worse than
                // the other way round.
                .keyboardShortcut("r", modifiers: [.command, .shift])
                Divider()
                // The same words as the button at the bottom of the sidebar. It
                // was "Start the queue" here and "Transcribe" in the window, and
                // the two did different things with nothing saying which.
                Button("Transcribe all") {
                    NotificationCenter.default.post(name: .scribaStart, object: nil)
                }
                .keyboardShortcut("t", modifiers: .command)
                Button("Stop") {
                    NotificationCenter.default.post(name: .scribaStop, object: nil)
                }
                .keyboardShortcut(".", modifiers: .command)
            }
            if WindowShot.isEnabled {
                CommandGroup(after: .saveItem) {
                    Button("Save window as PNG") { WindowShot.save() }
                        .keyboardShortcut("p", modifiers: [.command, .option])
                }
            }
        }

        Settings {
            SettingsView()
        }
    }
}

extension Notification.Name {
    /// Posted by the menu commands. The window listens; the menu items do not need
    /// to know anything about the queue.
    static let scribaRefresh = Notification.Name("dev.nerelli.scriba.refresh")
    static let scribaStart = Notification.Name("dev.nerelli.scriba.start")
    static let scribaStop = Notification.Name("dev.nerelli.scriba.stop")
    static let scribaRecord = Notification.Name("dev.nerelli.scriba.record")
}

/// Owns what has to outlive the window, and sees the app out.
///
/// The delegate is made once when the app starts and kept until it quits, which
/// is what the model needs: closing the window must not take the queue, the
/// engine or the recorder with it. Quitting has to take the engine with it, and
/// close a recording that is in progress.
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let model = AppModel()

    /// Opens the window (or brings it forward). Set by the App, which is the one
    /// that can read the action from the environment.
    var showWindow: (() -> Void)?

    /// A click on the Dock icon, or `open` on a running app, with no window on
    /// screen. The window scene does not come back by itself after it was closed:
    /// the system sends this and nothing else reopens it.
    func applicationShouldHandleReopen(_ sender: NSApplication,
                                       hasVisibleWindows flag: Bool) -> Bool {
        if !flag { showWindow?() }
        return true
    }

    /// Closing the window is not quitting. An app whose only scene is a single
    /// `Window` quits when that window closes unless it says otherwise, and the
    /// first build with one stopped a transcription halfway the moment its
    /// window was closed, which is the one thing the README promises it will not.
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }

    /// A recording is closed by an async stop, and the main thread cannot wait
    /// for it: the stop runs on the main actor, so waiting for it here is waiting
    /// for oneself, and three seconds later the app exited with the file still
    /// open. Say the app will answer later, and answer when the file is closed.
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        // The run goes at once, whatever else is still being closed.
        model.engine.terminateChild()
        guard model.recorder.isRecording else { return .terminateNow }
        Task {
            await model.stopRecording()
            answerQuit()
        }
        // A speech model that never finishes must not keep the app from quitting.
        Task {
            try? await Task.sleep(nanoseconds: 8_000_000_000)
            answerQuit()
        }
        return .terminateLater
    }

    private var answered = false

    private func answerQuit() {
        guard !answered else { return }
        answered = true
        NSApp.reply(toApplicationShouldTerminate: true)
    }

    func applicationWillTerminate(_ notification: Notification) {
        model.engine.terminateChild()
        model.letTheMacSleep()
    }
}

struct SettingsView: View {
    @State private var python = Engine.pythonPath
    @State private var engine = Engine.enginePath
    @AppStorage("liveTextEnabled") private var liveText = false
    @AppStorage(Engine.studioVoiceKey) private var studioVoice = false
    @AppStorage(Engine.studioVoiceIntensityKey) private var studioVoiceIntensity = 0.5
    @AppStorage(Engine.layerMixKey) private var layerMix = 0.5
    @State private var token = ""
    @State private var tokenState: TokenState = .unknown

    enum TokenState: Equatable {
        case unknown, present, saved, missing, failed(String)
        /// A token is there and the Keychain would not remove it.
        case stuck(String)
        /// The Keychain is being asked, with the word for what is happening.
        case working(String)
    }

    private var engineFolderState: String {
        var isDir: ObjCBool = false
        let there = FileManager.default.fileExists(atPath: engine, isDirectory: &isDir)
        if !there || !isDir.boolValue {
            return "No folder there. Leave it as it is if you installed scriba with "
                 + "pip: this only matters when you are running it from a clone."
        }
        return FileManager.default.fileExists(atPath: engine + "/scriba/cli.py")
            ? "Found." : "Folder found, but no scriba package inside it."
    }

    var body: some View {
        Form {
            Section("Engine") {
                TextField("Python interpreter", text: $python)
                    .onChange(of: python) { _, v in Engine.pythonPath = v }
                Text(FileManager.default.isExecutableFile(atPath: python)
                     ? "Found."
                     : "Not found. It has to be the python from the conda environment where whisperx is installed.")
                    .font(.caption)
                    .foregroundStyle(FileManager.default.isExecutableFile(atPath: python)
                                     ? Color.secondary : Color.red)

                TextField("scriba package folder", text: $engine)
                    .onChange(of: engine) { _, v in Engine.enginePath = v }
                // This one is the process working directory, so a wrong value
                // fails before Python is reached and the error blames the
                // interpreter. It used to be the field without validation, which
                // is the wrong way round.
                Text(engineFolderState)
                    .font(.caption)
                    .foregroundStyle(Engine.isEngineFolderUsable ? Color.secondary : Color.red)
            }
            Section("Recording") {
                Toggle("Show the words while recording", isOn: $liveText)
                Text("A preview from the speech model built into macOS 26, about a "
                     + "second behind the speaker and with its own punctuation. It is "
                     + "not kept: the document is still produced afterwards by "
                     + "whisper, which makes fewer mistakes. Set this before you "
                     + "start; it cannot be changed mid-recording.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Recordings from an iPhone") {
                Toggle("Studio Voice on Spatial Audio recordings", isOn: $studioVoice)
                Slider(value: $studioVoiceIntensity, in: 0...1) { Text("Intensity") }
                    .disabled(!studioVoice)
                Text("An iPhone that records Spatial Audio keeps the room in a track of "
                     + "its own. Studio Voice pushes it and its echo down and keeps the "
                     + "voices near the phone, as Voice Memos does when it plays them, "
                     + "and the transcript is made from that. Worth it where the room "
                     + "drowns the voices; on a clear recording it loses short replies. "
                     + "Needs macOS 26; other recordings are read as they are.")
                    .font(.caption).foregroundStyle(.secondary)
                Slider(value: $layerMix, in: 0...1) {
                    Text("Layer Mix")
                } minimumValueLabel: {
                    Text("First")
                } maximumValueLabel: {
                    Text("Second")
                }
                Text("A layered recording holds two takes, the second made over the "
                     + "first. Both are transcribed, weighed as this says.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("pyannote token") {
                // A secure field, and the value never leaves this window: it goes
                // to the `security` tool on its standard input, never in an
                // argument list or the environment, where other processes could
                // read it. The old instruction here was to type `scriba token
                // hf_...` into a terminal, which leaves the secret in the shell
                // history.
                SecureField("hf_…", text: $token)
                    .onSubmit(saveToken)
                    .disabled(busy)
                HStack {
                    Button("Save in the Keychain", action: saveToken)
                        .disabled(busy || token.trimmingCharacters(in: .whitespaces).isEmpty)
                    if tokenHere {
                        Button("Remove", action: removeToken)
                            .disabled(busy)
                    }
                    Spacer()
                }
                Text(tokenMessage)
                    .font(.caption)
                    .foregroundStyle(tokenColour)
            }
        }
        .formStyle(.grouped)
        .frame(width: 520, height: 640)
        .onAppear { tokenState = Keychain.hasToken() ? .present : .missing }
    }

    private var tokenHere: Bool {
        switch tokenState {
        case .present, .saved, .stuck: return true
        default: return false
        }
    }

    private var busy: Bool {
        if case .working = tokenState { return true }
        return false
    }

    /// Says "no token" only when the Keychain no longer has one. This used to
    /// change the message whatever the Keychain answered.
    ///
    /// Both of these wait for the `security` tool off the main thread: if the
    /// Keychain ever asks the person something, the window stays alive while it
    /// waits, and says so.
    private func removeToken() {
        guard !busy else { return }
        token = ""
        tokenState = .working("Removing the token…")
        Task {
            let why = await Keychain.inBackground { Keychain.forget() }
            tokenState = why.map { .stuck($0) } ?? .missing
        }
    }

    private func saveToken() {
        guard !busy else { return }
        let value = token
        tokenState = .working("Saving the token…")
        Task {
            let problem = await Keychain.inBackground { Keychain.save(value) }
            if let problem {
                tokenState = .failed(problem)
            } else {
                // Clear the field on success. Leaving the token sitting in a window
                // that anybody walking past can reveal is the thing this section
                // exists to avoid.
                token = ""
                tokenState = .saved
            }
        }
    }

    private var tokenMessage: String {
        switch tokenState {
        case .unknown: return ""
        case .present: return "A token is stored. Type a new one to replace it."
        case .saved: return "Saved. Diarization will work on the next run."
        case .missing:
            return "No token yet. Without one the recordings are transcribed but not "
                 + "separated by speaker. Get one from huggingface.co/settings/tokens "
                 + "and accept the conditions on pyannote/speaker-diarization-community-1."
        case .failed(let why), .stuck(let why), .working(let why): return why
        }
    }

    private var tokenColour: Color {
        switch tokenState {
        case .failed, .stuck: return .red
        case .saved, .present: return .secondary
        default: return .secondary
        }
    }
}
