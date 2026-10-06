import Foundation
import Testing

@testable import Scriba

/// How the engine's output becomes lines, and which line says why a run failed.
///
/// The reason on a failed row was once the tail of a wrapped sentence about memory
/// allocations. Three things can lose the marker line that carries the real one,
/// and each has a case here: a half line from one pipe glued to the next thing
/// from the other, a chunk that ends inside a character, and a reason that quotes
/// a library's warning.
struct PipeReadingTests {

    private static func data(_ text: String) -> Data { Data(text.utf8) }

    @Test("lines come out whole, and what has no newline yet waits for the next read")
    func linesAndResidue() {
        var buffer = LineBuffer()
        #expect(buffer.append(Self.data("first\nsec")) == ["first"])
        #expect(buffer.append(Self.data("ond\nthird")) == ["second"])
        #expect(buffer.finish() == ["third"])
    }

    @Test("a chunk boundary inside a character loses nothing")
    func boundaryInsideACharacter() {
        // "è" is two bytes and "€" three; the line has both and is cut inside each.
        let line = Array("[scriba:error] caffè costs 3€ today\n".utf8)
        for cut in 1..<line.count {
            var buffer = LineBuffer()
            let got = buffer.append(Data(line[..<cut])) + buffer.append(Data(line[cut...]))
            #expect(got == ["[scriba:error] caffè costs 3€ today"], "cut at \(cut)")
        }
    }

    @Test("a carriage return ends a line, so a redrawn progress bar does not hide what follows")
    func carriageReturn() {
        var buffer = LineBuffer()
        let got = buffer.append(Self.data("Progress: 40%\r[scriba:error] it broke\n"))
        #expect(got == ["Progress: 40%", "[scriba:error] it broke"])
    }

    @Test("a byte that is not UTF-8 costs one character, not the line or the chunk")
    func invalidBytes() {
        var buffer = LineBuffer()
        var bytes = Array("before ".utf8)
        bytes.append(0xFF)
        bytes += Array(" after\nnext\n".utf8)
        let got = buffer.append(Data(bytes))
        #expect(got.count == 2)
        #expect(got[0].hasPrefix("before "))
        #expect(got[0].hasSuffix(" after"))
        #expect(got[1] == "next")
    }

    @Test("the line without a newline is given up at the end of the stream")
    func finishGivesTheRest() {
        var buffer = LineBuffer()
        #expect(buffer.append(Self.data("one\ntwo")) == ["one"])
        #expect(buffer.finish() == ["two"])
        #expect(buffer.finish() == [])
    }

    @Test("an endless line is let go at a character boundary, not held for ever or cut in half")
    func longLine() {
        var buffer = LineBuffer()
        let chunk = String(repeating: "é", count: LineBuffer.longestLine)  // twice as many bytes
        let first = buffer.append(Self.data(chunk))
        #expect(first.count == 1)
        #expect(first[0].allSatisfy { $0 == "é" })
        #expect(!first[0].contains("\u{FFFD}"))
        // And what is left still joins up with the next read.
        let rest = buffer.append(Self.data("é\n"))
        #expect(rest.joined().allSatisfy { $0 == "é" })
    }

    // MARK: the marker

    @Test("the reason is what follows the marker")
    func reasonAfterMarker() {
        #expect(Engine.failureReason(inLine: "[scriba:error] could not load the model: MPS backend out of memory (allocated 4 GiB)")
                == "could not load the model: MPS backend out of memory (allocated 4 GiB)")
    }

    @Test("a missing token prints both markers on one line and both come out")
    func tokenMarkerInsideTheErrorLine() {
        let line = "[scriba:error] \(Engine.errNoToken) Add a Hugging Face token in Settings."
        #expect(Engine.failureReason(inLine: line) == "Add a Hugging Face token in Settings.")
    }

    @Test("a marker that is not at the start of the line is still found")
    func markerBehindSomething() {
        #expect(Engine.failureReason(inLine: "  41%|████ | [scriba:error] disk full") == "disk full")
    }

    @Test("ordinary output, and a marker with nothing after it, are not reasons")
    func notReasons() {
        #expect(Engine.failureReason(inLine: "Progress: 50%") == nil)
        #expect(Engine.failureReason(inLine: "1 of 1 did not go through:") == nil)
        #expect(Engine.failureReason(inLine: "[scriba:error]") == nil)
        #expect(Engine.failureReason(inLine: "[scriba:error]   ") == nil)
    }

    @Test("a message of several lines becomes one for a row")
    func oneLine() {
        #expect(Engine.oneLine("could not load X:\n  MPS backend out of memory\n\n") ==
                "could not load X: MPS backend out of memory")
        #expect(Engine.oneLine("") == "")
    }
}

/// What Save sends, and how a name is taken back.
struct NameSavingTests {

    @Test("a typed name goes as it is, trimmed")
    func typed() {
        #expect(Engine.namesToSend(drafts: ["SPEAKER_00": "  Ada "], stored: [:]) == ["SPEAKER_00": "Ada"])
    }

    @Test("a stored name wiped from its field goes as an empty name")
    func cleared() {
        let send = Engine.namesToSend(drafts: ["SPEAKER_00": "Ada", "SPEAKER_01": ""],
                                      stored: ["SPEAKER_00": "Ada", "SPEAKER_01": "Rafiq"])
        #expect(send == ["SPEAKER_00": "Ada", "SPEAKER_01": ""])
    }

    @Test("a field of spaces counts as cleared")
    func spaces() {
        #expect(Engine.namesToSend(drafts: ["SPEAKER_01": "   "], stored: ["SPEAKER_01": "Rafiq"])
                == ["SPEAKER_01": ""])
    }

    @Test("an empty field with nothing stored has nothing to remove, and with every field so there is nothing to save")
    func nothingToRemove() {
        #expect(Engine.namesToSend(drafts: ["SPEAKER_00": "", "SPEAKER_01": ""], stored: [:]).isEmpty)
        #expect(Engine.namesToSend(drafts: ["SPEAKER_00": ""], stored: ["SPEAKER_00": ""]).isEmpty)
    }

    @Test("the command carries LABEL= for an empty name, in a fixed order")
    func arguments() {
        let args = Engine.nameArguments(["SPEAKER_01": "", "SPEAKER_00": "Ada"])
        #expect(args == ["SPEAKER_00=Ada", "SPEAKER_01="])
    }
}

/// A recording must never be written over one that exists.
struct RecordingNameTests {

    private static func folder() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("scriba-names-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    @Test("two recordings with the same stamp get different files, and the first is left alone")
    func sameSecond() throws {
        let dir = try Self.folder(); defer { try? FileManager.default.removeItem(at: dir) }
        let first = try Recorder.reserve(in: dir, stamp: "2026-10-06 14-30-05")
        try Data("the first recording".utf8).write(to: first)
        let second = try Recorder.reserve(in: dir, stamp: "2026-10-06 14-30-05")
        let third = try Recorder.reserve(in: dir, stamp: "2026-10-06 14-30-05")

        #expect(first.lastPathComponent == "Recording 2026-10-06 14-30-05.wav")
        #expect(second.lastPathComponent == "Recording 2026-10-06 14-30-05 2.wav")
        #expect(third.lastPathComponent == "Recording 2026-10-06 14-30-05 3.wav")
        #expect(try String(contentsOf: first, encoding: .utf8) == "the first recording")
    }

    @Test("a stamp nobody used is taken as it is")
    func freshStamp() throws {
        let dir = try Self.folder(); defer { try? FileManager.default.removeItem(at: dir) }
        _ = try Recorder.reserve(in: dir, stamp: "2026-10-06 14-30-05")
        let next = try Recorder.reserve(in: dir, stamp: "2026-10-06 14-30-06")
        #expect(next.lastPathComponent == "Recording 2026-10-06 14-30-06.wav")
    }

    @Test("a folder that cannot be written to is an error, not a loop")
    func unwritable() {
        let nowhere = URL(fileURLWithPath: "/no/such/folder/\(UUID().uuidString)")
        #expect(throws: (any Error).self) {
            try Recorder.reserve(in: nowhere, stamp: "2026-10-06 14-30-05")
        }
    }
}

/// What Settings says when the Keychain would not remove the token. Only the words:
/// nothing here reaches the Keychain.
struct RemovalMessageTests {
    static let terminal = "security delete-generic-password -s scriba-hf-token"

    @Test("a refusal names the terminal command that does work")
    func refusal() {
        let ran = Keychain.Ran(status: 1, output: "security: User interaction is not allowed.\n",
                               timedOut: false)
        let text = Keychain.removalProblem(ran, service: "scriba-hf-token")
        #expect(text.contains(Self.terminal))
        #expect(text.contains("different program"))
    }

    @Test("a delete that said yes while a token remains is not reported as removed")
    func saidYesButStillThere() {
        let ran = Keychain.Ran(status: 0, output: "", timedOut: false)
        let text = Keychain.removalProblem(ran, service: "scriba-hf-token")
        #expect(text.contains("still there"))
        #expect(text.contains(Self.terminal))
    }

    @Test("any other failure is told in the tool's own words, or by its status")
    func otherStatus() {
        let said = Keychain.removalProblem(Keychain.Ran(status: 36, output: "Something odd.\n",
                                                        timedOut: false))
        #expect(said.contains("Something odd."))
        let silent = Keychain.removalProblem(Keychain.Ran(status: 36, output: "", timedOut: false))
        #expect(silent.contains("status 36"))
    }

    @Test("a question nobody answered, and a tool that would not start, each get their own words")
    func timedOutAndMissing() {
        let late = Keychain.removalProblem(Keychain.Ran(status: 15, output: "", timedOut: true))
        #expect(late.contains("did not answer"))
        #expect(late.contains(Self.terminal))
        let missing = Keychain.removalProblem(nil)
        #expect(missing.contains("could not run"))
        #expect(missing.contains(Self.terminal))
    }
}

/// What is told to `security -i`, and the machinery that tells it. The commands
/// are built by pure functions so they can be read here; none of these runs one.
/// What does run is the plumbing, against tools that touch nothing.
struct KeychainCommandTests {
    static let token = "hf_AdaLakeTrip0123456789abcdefABCDEF"

    // MARK: the token

    @Test("a Hugging Face token is accepted: hf_ and then letters and digits")
    func acceptsATokenShapedLikeOne() {
        #expect(Keychain.tokenProblem(Self.token) == nil)
        #expect(Keychain.tokenProblem("hf_0") == nil)
    }

    @Test("anything else is refused with a message, and the message never repeats the token")
    func refusesWhatWouldBreakTheLine() {
        let bad = ["", "abc123", "hf_", "HF_abc", "hf_ab cd", "hf_ab\"cd", "hf_ab\\cd",
                   "hf_ab'cd", "hf_ab;rm", "hf_ab\ncd", "hf_ab\tcd", "hf_ab-cd", "hf_ab_cd",
                   "hf_Zoë", "hf_ab$(x)", "hf_ab`x`", " hf_abc", "hf_abc "]
        for token in bad {
            let message = Keychain.tokenProblem(token)
            #expect(message != nil, "accepted \(token.debugDescription)")
            if let message, token.count > 3 { #expect(!message.contains(token)) }
        }
    }

    // MARK: the command text

    @Test("the add command is the one `scriba token` makes, on one line, in quotes")
    func addCommandText() {
        let line = Keychain.addCommand(token: Self.token, service: "scriba-hf-token", account: "ada")
        #expect(line == "add-generic-password -U -s \"scriba-hf-token\" -a \"ada\" "
                      + "-w \"\(Self.token)\"\n")
    }

    @Test("the delete command names the service and nothing else")
    func deleteCommandText() {
        #expect(Keychain.deleteCommand(service: "scriba-hf-token")
                == "delete-generic-password -s \"scriba-hf-token\"\n")
    }

    @Test("a token that is not safe gives no command at all")
    func noCommandForABadToken() {
        #expect(Keychain.addCommand(token: "hf_ab cd", service: "s", account: "ada") == nil)
        #expect(Keychain.addCommand(token: "hf_abc\nadd-generic-password -w x", service: "s",
                                    account: "ada") == nil)
    }

    @Test("a name with a quote or a backslash is escaped, one with a line break is refused")
    func namesAreQuoted() {
        #expect(Keychain.quoted("Rafiq \"R\" Ada") == "\"Rafiq \\\"R\\\" Ada\"")
        #expect(Keychain.quoted("a\\b") == "\"a\\\\b\"")
        #expect(Keychain.quoted("ada\nbob") == nil)
        #expect(Keychain.quoted("ada\u{7}") == nil)
        #expect(Keychain.addCommand(token: Self.token, service: "s", account: "ada\nbob") == nil)
        #expect(Keychain.deleteCommand(service: "s\r") == nil)
    }

    @Test("whatever the names are, the command is a single line")
    func singleLine() throws {
        for account in ["ada", "Rafiq R", "o'brien", "a\"b", "a\\b"] {
            let line = try #require(Keychain.addCommand(token: Self.token, service: "svc",
                                                        account: account))
            #expect(line.filter(\.isNewline).count == 1)
            #expect(line.hasSuffix("\n"))
        }
    }

    @Test("the account is the login name the engine writes with")
    func accountIsNotEmpty() {
        #expect(!Keychain.account.isEmpty)
    }

    // MARK: what the tool said

    @Test("bookkeeping lines are not the reason, and the token is taken out of what is left")
    func readingTheOutput() {
        let output = "add-generic-password: returned 45\n"
                   + "security: SecKeychainItemCreateFromContent: already exists \(Self.token)\n"
        let said = Keychain.reason(in: output, removing: Self.token)
        #expect(said == "security: SecKeychainItemCreateFromContent: already exists …")
        #expect(Keychain.reason(in: "add-generic-password: returned 1\n") == nil)
        #expect(Keychain.reason(in: "") == nil)
    }

    @Test("a save that worked says nothing, and one that did not says why")
    func saveProblems() {
        let ok = Keychain.Ran(status: 0, output: "", timedOut: false)
        #expect(Keychain.saveProblem(ok, token: Self.token) == nil)

        let busy = Keychain.Ran(status: 1, output: "security: something else\n", timedOut: false)
        let said = Keychain.saveProblem(busy, token: Self.token) ?? ""
        #expect(said.contains("something else"))
        #expect(!said.contains("security delete-generic-password"))

        let duplicate = Keychain.Ran(status: 45,
                                     output: "security: the item already exists \(Self.token)\n",
                                     timedOut: false)
        let hint = Keychain.saveProblem(duplicate, token: Self.token) ?? ""
        #expect(hint.contains("security delete-generic-password -s scriba-hf-token"))
        #expect(!hint.contains(Self.token))

        let silent = Keychain.saveProblem(Keychain.Ran(status: 7, output: "", timedOut: false),
                                          token: Self.token) ?? ""
        #expect(silent.contains("status 7"))

        let late = Keychain.saveProblem(Keychain.Ran(status: 15, output: "", timedOut: true),
                                        token: Self.token) ?? ""
        #expect(late.contains("did not answer"))
        #expect(Keychain.saveProblem(nil, token: Self.token)?.contains("could not run") == true)
    }

    // MARK: the plumbing, against tools that touch nothing

    @Test("what is written goes to the child's standard input, whole")
    func inputReachesTheChild() throws {
        let ran = try #require(Keychain.run(tool: "/bin/cat", arguments: [],
                                            input: "line one\nline two\n"))
        #expect(ran.status == 0)
        #expect(!ran.timedOut)
        #expect(ran.output == "line one\nline two\n")
    }

    @Test("the secret is not among the arguments the child was started with")
    func secretIsNotAnArgument() throws {
        // `ps` shows a process's arguments to everybody. This child prints its own.
        let ran = try #require(Keychain.run(tool: "/bin/sh",
                                            arguments: ["-c", "ps -o args= -p $$; cat >/dev/null"],
                                            input: Self.token))
        #expect(ran.status == 0)
        #expect(ran.output.contains("/bin/sh"))
        #expect(!ran.output.contains(Self.token))
        // And the default arguments of the real call are the interactive flag alone.
        #expect(Keychain.securityTool == "/usr/bin/security")
    }

    @Test("a status that is not zero is passed on, with what was said")
    func failureIsReported() throws {
        let ran = try #require(Keychain.run(tool: "/bin/sh", arguments: ["-c", "echo no >&2; exit 3"],
                                            input: ""))
        #expect(ran.status == 3)
        #expect(ran.output == "no\n")
    }

    @Test("a child that never answers is ended at the limit")
    func limit() throws {
        let started = Date()
        let ran = try #require(Keychain.run(tool: "/bin/sleep", arguments: ["30"], input: "",
                                            limit: 0.3))
        #expect(ran.timedOut)
        #expect(Date().timeIntervalSince(started) < 10)
    }

    @Test("a child that has already gone does not take this process down when written to")
    func writingToAGoneChild() throws {
        let big = String(repeating: "x", count: 1_000_000)
        let ran = try #require(Keychain.run(tool: "/usr/bin/true", arguments: [], input: big))
        #expect(ran.status == 0)
    }

    @Test("a tool that is not there is nil, not a crash")
    func missingTool() {
        #expect(Keychain.run(tool: "/nonexistent/tool", arguments: [], input: "") == nil)
    }

    @Test("the real tool reads its commands from a pipe: a command that touches no keychain")
    func securityReadsAPipe() throws {
        let ran = try #require(Keychain.run(input: "help\n", limit: 20))
        #expect(ran.status == 0)
        #expect(ran.output.contains("add-generic-password"))
    }
}

/// The sentence that says the live text is guessing its language.
struct LiveLanguageNoteTests {

    @Test("detect automatically means the language of this Mac, and the pane says so")
    func guess() {
        let note = liveLanguageNote(language: "auto", systemCode: "it")
        #expect(note?.contains("Italian") == true)
        #expect(note?.contains("guess") == true)
    }

    @Test("a language the person chose is not a guess")
    func chosen() {
        #expect(liveLanguageNote(language: "es", systemCode: "it") == nil)
    }

    @Test("with no system language it falls back to English, as the transcriber does")
    func fallback() {
        #expect(liveLanguageNote(language: "auto", systemCode: nil)?.contains("English") == true)
    }
}
