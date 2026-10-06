import Foundation
import Security

/// The pyannote token, in the Keychain, under the same name the engine reads.
///
/// The Python side looks for a generic password with service `scriba-hf-token`
/// by running `security find-generic-password -s scriba-hf-token -w` and takes
/// what it prints (config.py, `keychain_get`). It does that after a whole
/// transcription, when nobody is at the machine, so the item has to be one that
/// `security` may read without asking.
///
/// An item this application made through the Security framework is not. It
/// belongs to its creator: the trusted-application list can name `security`,
/// but on the login keychain the item also carries a partition list, stamped
/// with the creator's identity, and for an ad-hoc signed application that is a
/// hash that changes with every build. `security` is an Apple tool in another
/// partition, so its read raised the keychain password dialog in the middle of
/// the run, and denying it made the run fail as "token missing". Nothing in
/// this process can add a partition: that takes the keychain password.
///
/// So the item is made by the program that reads it. Settings starts
/// `/usr/bin/security -i` and writes the command on its standard input, which
/// makes exactly the item `scriba token` makes. The secret goes down a pipe: it
/// is never in an argument list, where any other process on the machine can
/// read it out of `ps` for as long as the command runs, and never in the
/// environment, which the same user's other processes can read too.
enum Keychain {
    static let service = "scriba-hf-token"

    /// The tool the engine reads the token with, and the one that writes it here.
    static let securityTool = "/usr/bin/security"

    /// The account `scriba token` writes with (config.py, `keychain_set`:
    /// `os.getlogin()`, which is getlogin(3)). The same account is what lets
    /// `-U` replace that item rather than add a second one under another name,
    /// which `find-generic-password -s` could answer with first.
    static var account: String {
        if let name = getlogin() {
            let text = String(cString: name)
            if !text.isEmpty { return text }
        }
        return NSUserName()
    }

    /// The service name is a parameter so the tests can round-trip against a
    /// throwaway one. The real entry holds the user's token and nothing automated
    /// should be writing over it to prove that writing works.
    private static func query(_ service: String) -> [String: Any] {
        [kSecClass as String: kSecClassGenericPassword,
         kSecAttrService as String: service]
    }

    /// Whether a token is there. Deliberately does not return it: nothing in this
    /// application needs to see the value, only whether one has been set. Asking
    /// for no data and no attributes is asking for nothing the item guards, so
    /// this never raises a dialog, whoever made the item.
    static func hasToken(service: String = service) -> Bool {
        var q = query(service)
        q[kSecMatchLimit as String] = kSecMatchLimitOne
        return SecItemCopyMatching(q as CFDictionary, nil) == errSecSuccess
    }

    // MARK: - what is told to `security`

    /// Nil when `token` can go on a command line without being changed, else what
    /// to tell the person. A Hugging Face token is `hf_` and then letters and
    /// digits, nothing else, and that is all that is accepted: a space, a quote
    /// or a backslash from a bad paste would be read by `security` as the end of
    /// the value or as the start of another argument, and what got stored would
    /// be a different token from the one typed, which fails much later as a 401
    /// that explains nothing. The message never repeats what was typed.
    static func tokenProblem(_ token: String) -> String? {
        if token.isEmpty { return "The token is empty." }
        guard token.hasPrefix("hf_"), token.count > 3 else {
            return "A Hugging Face token starts with hf_ and has more after it. Copy it "
                 + "again from huggingface.co/settings/tokens."
        }
        let allowed = Set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        guard token.dropFirst(3).allSatisfy({ allowed.contains($0) }) else {
            return "A Hugging Face token is hf_ followed by letters and digits only, and "
                 + "this has something else in it, such as a space or a quote. Nothing "
                 + "was saved. Copy the token again from huggingface.co/settings/tokens."
        }
        return nil
    }

    /// One argument as `security -i` reads it: in double quotes, with a quote or
    /// a backslash inside escaped. Nil when it holds a line break or another
    /// control character, which would end the command or be read as something
    /// else.
    static func quoted(_ text: String) -> String? {
        var out = "\""
        for scalar in text.unicodeScalars {
            if scalar.value < 0x20 || scalar.value == 0x7f { return nil }
            if scalar == "\"" || scalar == "\\" { out += "\\" }
            out.unicodeScalars.append(scalar)
        }
        return out + "\""
    }

    /// The line that stores the token: the one `scriba token` stores, with `-U`
    /// so a token already there is replaced. Nil if the token is not one that
    /// can be written safely, or a name has a control character in it.
    static func addCommand(token: String, service: String, account: String) -> String? {
        guard tokenProblem(token) == nil,
              let s = quoted(service), let a = quoted(account), let t = quoted(token)
        else { return nil }
        return "add-generic-password -U -s \(s) -a \(a) -w \(t)\n"
    }

    /// The line that removes one item stored under `service`.
    static func deleteCommand(service: String) -> String? {
        guard let s = quoted(service) else { return nil }
        return "delete-generic-password -s \(s)\n"
    }

    // MARK: - running it

    /// What a child process did.
    struct Ran: Equatable {
        var status: Int32
        /// Standard output and standard error together.
        var output: String
        /// It was still running at the limit and was ended.
        var timedOut: Bool
    }

    /// Run `tool`, write `input` on its standard input, and wait for it. Nil when
    /// it could not be started at all.
    ///
    /// Blocks, so it is called off the main thread. It waits for a limit rather
    /// than for ever because the Keychain may put a question to the person, and
    /// a question nobody answers must not keep a button spinning.
    static func run(tool: String = securityTool, arguments: [String] = ["-i"],
                    input: String, limit: TimeInterval = 60) -> Ran? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: tool)
        process.arguments = arguments
        let stdin = Pipe()
        let out = Pipe()
        process.standardInput = stdin
        process.standardOutput = out
        process.standardError = out
        let done = DispatchSemaphore(value: 0)
        process.terminationHandler = { _ in done.signal() }
        do { try process.run() } catch { return nil }

        // A tool that has already gone must not take this application down with
        // it: writing to a pipe nobody reads raises SIGPIPE, which ends the
        // process. Ask for the error instead, on this one descriptor.
        _ = fcntl(stdin.fileHandleForWriting.fileDescriptor, F_SETNOSIGPIPE, 1)
        try? stdin.fileHandleForWriting.write(contentsOf: Data(input.utf8))
        try? stdin.fileHandleForWriting.close()

        var timedOut = false
        if done.wait(timeout: .now() + limit) == .timedOut {
            timedOut = true
            process.terminate()
            _ = done.wait(timeout: .now() + 5)
        }
        let data = out.fileHandleForReading.readDataToEndOfFile()
        return Ran(status: process.terminationStatus,
                   output: String(decoding: data, as: UTF8.self), timedOut: timedOut)
    }

    /// The first thing `security` said that is not bookkeeping, with the token
    /// taken out of it in case the tool ever quotes what it was given.
    static func reason(in output: String, removing token: String = "") -> String? {
        for raw in output.split(whereSeparator: \.isNewline) {
            var line = raw.trimmingCharacters(in: .whitespaces)
            // "add-generic-password: returned 45" is the tool counting, not saying.
            if line.isEmpty || line.range(of: #": returned \d+$"#, options: .regularExpression) != nil {
                continue
            }
            if !token.isEmpty { line = line.replacingOccurrences(of: token, with: "…") }
            return line
        }
        return nil
    }

    // MARK: - saving and removing

    /// `work` on a thread of its own, for the caller to await. Saving and removing
    /// wait for a child process, and for a person if the Keychain asks one a
    /// question, which is not something to do on the main thread or to park on
    /// the pool Swift's tasks share.
    static func inBackground<T>(_ work: @escaping @Sendable () -> T) async -> T {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async { continuation.resume(returning: work()) }
        }
    }

    /// Store, replacing whatever was there. Returns nil on success, a message otherwise.
    /// Blocks while `security` runs: call it off the main thread.
    static func save(_ token: String, service: String = service) -> String? {
        let value = token.trimmingCharacters(in: .whitespacesAndNewlines)
        if let problem = tokenProblem(value) { return problem }
        guard let command = addCommand(token: value, service: service, account: account) else {
            return "The Keychain could not be told to store this token, so nothing was saved."
        }
        let ran = run(input: command)
        if let problem = saveProblem(ran, token: value, service: service) { return problem }
        // Whatever it said, an item has to be there now.
        guard hasToken(service: service) else {
            return "The Keychain said it saved the token, but none is there. Try again, "
                 + "or save it from a terminal:\nscriba token hf_…"
        }
        return nil
    }

    /// What to tell somebody whose token was not stored, or nil if it was.
    static func saveProblem(_ ran: Ran?, token: String, service: String = service) -> String? {
        guard let ran else {
            return "Scriba could not run \(securityTool), so nothing was saved."
        }
        if ran.timedOut {
            return "The Keychain did not answer within a minute. If a window is asking for "
                 + "the login keychain password, answer it or cancel it, then save again."
        }
        guard ran.status != 0 else { return nil }
        let said = reason(in: ran.output, removing: token)
        var text = said.map { "The Keychain did not take the token: \($0)" }
            ?? "The Keychain did not take the token (the security tool stopped with "
             + "status \(ran.status))."
        if blockedByAnotherProgram(ran.output) {
            text += "\nAn older token may be in the way, put there by a different program. "
                  + "Remove it from a terminal, then save again:\n"
                  + "security delete-generic-password -s \(service)"
        }
        return text
    }

    /// Whether the tool's words say an item exists that it may not change.
    private static func blockedByAnotherProgram(_ output: String) -> Bool {
        let text = output.lowercased()
        return ["already exists", "not allowed", "denied", "cancel", "authoriz"]
            .contains { text.contains($0) }
    }

    /// Remove the token, and say what the Keychain did. Nil when no token is
    /// left, a sentence when one is. The answer is whether the item is still
    /// there, not the status of the delete: the status was ignored, and the
    /// panel announced that the token was gone while the engine went on using it.
    /// Blocks while `security` runs: call it off the main thread.
    @discardableResult
    static func forget(service: String = service) -> String? {
        guard hasToken(service: service) else { return nil }
        guard let command = deleteCommand(service: service) else {
            return removalProblem(nil, service: service)
        }
        var last: Ran?
        // One delete takes one item. A second item under the same service, from
        // another account, would go on answering the engine, so go on while the
        // tool says it removed one and one is still there.
        for _ in 0..<4 {
            last = run(input: command)
            guard let ran = last, !ran.timedOut, ran.status == 0, hasToken(service: service)
            else { break }
        }
        guard hasToken(service: service) else { return nil }
        return removalProblem(last, service: service)
    }

    /// What to tell somebody whose token is still in the Keychain after a delete.
    static func removalProblem(_ ran: Ran?, service: String = service) -> String {
        let terminal = "security delete-generic-password -s \(service)"
        guard let ran else {
            return "Scriba could not run \(securityTool), and the token is still there. "
                 + "Remove it from a terminal:\n\(terminal)"
        }
        if ran.timedOut {
            return "The Keychain did not answer within a minute, and the token is still "
                 + "there. If a window is asking for the login keychain password, answer "
                 + "it or cancel it, then try again. Or remove it from a terminal:\n\(terminal)"
        }
        if ran.status == 0 {
            // Said it was gone and it is not: another item answers to the name.
            return "The Keychain says it removed the token, but one is still there. "
                 + "Remove it from a terminal:\n\(terminal)"
        }
        if blockedByAnotherProgram(ran.output) {
            return "The Keychain did not let the security tool remove the token, because "
                 + "a different program put it there. Remove it from a terminal:\n\(terminal)"
        }
        let said = reason(in: ran.output) ?? "the security tool stopped with status \(ran.status)"
        return "The Keychain refused: \(said), and the token is still there. "
             + "Remove it from a terminal:\n\(terminal)"
    }
}
