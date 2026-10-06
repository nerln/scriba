import AppKit
import SwiftUI
import Testing
import Vision

@testable import Scriba

/// A window that can be put where nobody can see it: the views are laid out and
/// drawn as on screen, and nothing flashes up on the desktop of whoever runs the tests.
final class OffscreenWindow: NSWindow {
    override func constrainFrameRect(_ frameRect: NSRect, to screen: NSScreen?) -> NSRect { frameRect }
}

/// How the welcome screen lays out at the widths a window can give it.
///
/// After a relaunch the Language picker was 32 points wide and showed one letter:
/// a pop-up button offered less than its own width does not wrap or clip, it
/// shrinks to an ellipsis. Nothing here can ask a SwiftUI picker how wide it is, so
/// the screen is drawn and read, which is also what a person does.
@MainActor
struct WelcomeLayoutTests {
    static let languages = [("auto", "Detect automatically"), ("it", "Italian"),
                            ("en", "English"), ("es", "Spanish")]

    /// Welcome drawn at `width`, off screen.
    static func draw(width: CGFloat, height: CGFloat = 700, language: String = "auto") -> CGImage? {
        _ = NSApplication.shared
        NSApp.setActivationPolicy(.prohibited)
        let host = NSHostingView(rootView: Welcome(processed: 3, language: .constant(language),
                                                   languages: languages, onRecord: {}))
        let frame = NSRect(x: -20000, y: -20000, width: width, height: height)
        let window = OffscreenWindow(contentRect: frame, styleMask: [.titled, .resizable],
                                     backing: .buffered, defer: false)
        // Light, so the words are dark on white whatever the machine is set to.
        window.appearance = NSAppearance(named: .aqua)
        window.contentView = host
        window.setFrame(frame, display: true)
        window.orderFrontRegardless()
        defer { window.orderOut(nil) }
        RunLoop.current.run(until: Date().addingTimeInterval(0.4))
        host.layoutSubtreeIfNeeded()
        guard let rep = host.bitmapImageRepForCachingDisplay(in: host.bounds) else { return nil }
        host.cacheDisplay(in: host.bounds, to: rep)
        return rep.cgImage
    }

    /// The lines of text that can be read in the picture.
    static func read(_ image: CGImage) throws -> [String] {
        // The drawing has no background of its own: on white, as it is on screen.
        guard let context = CGContext(data: nil, width: image.width, height: image.height,
                                      bitsPerComponent: 8, bytesPerRow: 0,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return [] }
        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: image.width, height: image.height))
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        guard let flattened = context.makeImage() else { return [] }
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = false
        try VNImageRequestHandler(cgImage: flattened).perform([request])
        return (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }
    }

    @Test("the Language picker shows its whole name at every width the pane can have, side by side or stacked")
    func pickerKeepsItsWidth() throws {
        for width in [300.0, 360, 440, 520, 600, 900] {
            let image = try #require(Self.draw(width: width), "no picture at \(width)")
            let lines = try Self.read(image).joined(separator: "\n")
            #expect(lines.contains("Detect automatically"), "squeezed at \(width): \(lines)")
            #expect(lines.contains("Language"), "label lost at \(width)")
            // And the checkbox keeps its words, in one piece or wrapped, not cut to the box alone.
            #expect(lines.contains("recording"), "checkbox words lost at \(width)")
        }
    }

    @Test("a language that was picked is the one the picker shows")
    func showsTheChosenLanguage() throws {
        let image = try #require(Self.draw(width: 520, language: "es"))
        #expect(try Self.read(image).joined(separator: "\n").contains("Spanish"))
    }
}
