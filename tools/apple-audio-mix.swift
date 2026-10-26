// The Audio Mix of an iPhone recording made with Spatial Audio, rendered to a WAV.
//
// An iPhone Pro, and Voice Memos on it since iOS 26, can record Spatial Audio: the
// file holds a stereo track for everything that cannot play the rest, and four
// channels of Apple's positional audio with the analysis the phone made while
// recording. Photos and Voice Memos use that to offer Audio Mix (Standard, In-frame,
// Studio, Cinematic). The Cinematic framework does the same on a Mac, macOS 26 or
// later; ffmpeg cannot decode the positional track at all.
//
//   apple-audio-mix info <file>
//   apple-audio-mix render <file> <out.wav> <style> [intensity] [start] [seconds]
//
// `info` prints JSON. `render` writes 16-bit stereo PCM at 16 kHz with the effect
// burned in, the rate scriba works at; it then converts it like any other audio.
// Built by scriba/spatial.py the first time it is needed.

import AVFoundation
import Cinematic
import Foundation

let styles: [String: CNSpatialAudioRenderingStyle] = [
    "standard": .standard,
    "cinematic": .cinematic,
    "studio": .studio,
    "in-frame": .inFrame,
    // The foreground alone, as Studio hears it: the voices near the phone,
    // without the room.
    "voices": .studioForegroundStem,
]

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

func styleName(_ style: CNSpatialAudioRenderingStyle) -> String {
    styles.first { $0.value == style }?.key ?? "other"
}

@main
struct AppleAudioMix {
    static func main() async {
        let args = CommandLine.arguments
        guard args.count >= 3, ["info", "render"].contains(args[1]) else {
            fail("usage: apple-audio-mix info <file> | render <file> <out.wav> <style> "
                 + "[intensity] [start] [seconds]")
        }
        let asset = AVURLAsset(url: URL(fileURLWithPath: args[2]))
        let spatial = await CNAssetSpatialAudioInfo.assetContainsSpatialAudio(asset: asset)

        if args[1] == "info" {
            var out: [String: Any] = ["supported": CNAssetSpatialAudioInfo.isSupported,
                                      "spatial": spatial]
            if spatial, let info = try? await CNAssetSpatialAudioInfo(asset: asset) {
                out["default_style"] = styleName(info.defaultRenderingStyle)
                out["default_intensity"] = Double(info.defaultEffectIntensity)
            }
            let data = try! JSONSerialization.data(withJSONObject: out, options: [.sortedKeys])
            print(String(data: data, encoding: .utf8)!)
            return
        }

        guard args.count >= 5 else { fail("render needs <out.wav> <style>") }
        guard let style = styles[args[4]] else {
            fail("unknown style \(args[4]); one of \(styles.keys.sorted().joined(separator: ", "))")
        }
        guard CNAssetSpatialAudioInfo.isSupported else { fail("this Mac cannot render Audio Mix") }
        guard spatial else { fail("this file was not recorded with Spatial Audio") }
        let info: CNAssetSpatialAudioInfo
        do { info = try await CNAssetSpatialAudioInfo(asset: asset) } catch {
            fail("cannot read the Spatial Audio of this file: \(error.localizedDescription)")
        }
        let intensity = args.count > 5 ? Float(args[5]) ?? info.defaultEffectIntensity
                                       : info.defaultEffectIntensity
        do {
            try render(asset: asset, info: info, style: style, intensity: intensity,
                       to: URL(fileURLWithPath: args[3]),
                       start: args.count > 6 ? Double(args[6]) : nil,
                       seconds: args.count > 7 ? Double(args[7]) : nil)
        } catch {
            fail("rendering failed: \(error.localizedDescription)")
        }
    }

    static func render(asset: AVAsset, info: CNAssetSpatialAudioInfo,
                       style: CNSpatialAudioRenderingStyle, intensity: Float, to out: URL,
                       start: Double?, seconds: Double?) throws {
        var settings = info.assetReaderOutputSettings(for: .stereo)
        // Plain interleaved 16-bit, which any WAV reader takes, at the rate the
        // transcriber works at: an hour at 48 kHz is 690 MB of scratch file.
        settings[AVFormatIDKey] = kAudioFormatLinearPCM
        settings[AVSampleRateKey] = 16_000
        settings[AVLinearPCMBitDepthKey] = 16
        settings[AVLinearPCMIsFloatKey] = false
        settings[AVLinearPCMIsBigEndianKey] = false
        settings[AVLinearPCMIsNonInterleaved] = false
        let rate = (settings[AVSampleRateKey] as? NSNumber)?.doubleValue ?? 16_000
        let channels = (settings[AVNumberOfChannelsKey] as? NSNumber)?.intValue ?? 2

        let reader = try AVAssetReader(asset: asset)
        if let start {
            let begin = CMTime(seconds: start, preferredTimescale: 48_000)
            let length = seconds.map { CMTime(seconds: $0, preferredTimescale: 48_000) } ?? .positiveInfinity
            reader.timeRange = CMTimeRange(start: begin, duration: length)
        }
        let output = AVAssetReaderAudioMixOutput(audioTracks: [info.defaultSpatialAudioTrack],
                                                 audioSettings: settings)
        output.audioMix = info.audioMix(effectIntensity: intensity, renderingStyle: style)
        reader.add(output)
        guard reader.startReading() else {
            throw reader.error ?? CocoaError(.fileReadUnknown)
        }

        FileManager.default.createFile(atPath: out.path, contents: nil)
        let file = try FileHandle(forWritingTo: out)
        defer { try? file.close() }
        file.write(header(rate: Int(rate), channels: channels, bytes: 0))
        var written = 0
        while let buffer = output.copyNextSampleBuffer() {
            guard let block = CMSampleBufferGetDataBuffer(buffer) else { continue }
            let length = CMBlockBufferGetDataLength(block)
            var data = Data(count: length)
            data.withUnsafeMutableBytes { raw in
                _ = CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: length,
                                               destination: raw.baseAddress!)
            }
            file.write(data)
            written += length
        }
        if reader.status == .failed { throw reader.error ?? CocoaError(.fileReadUnknown) }
        try file.seek(toOffset: 0)
        file.write(header(rate: Int(rate), channels: channels, bytes: written))
    }

    static func header(rate: Int, channels: Int, bytes: Int) -> Data {
        var d = Data()
        func u32(_ v: Int) { var x = UInt32(v).littleEndian; d.append(Data(bytes: &x, count: 4)) }
        func u16(_ v: Int) { var x = UInt16(v).littleEndian; d.append(Data(bytes: &x, count: 2)) }
        d.append("RIFF".data(using: .ascii)!); u32(36 + bytes)
        d.append("WAVEfmt ".data(using: .ascii)!); u32(16)
        u16(1); u16(channels); u32(rate); u32(rate * channels * 2); u16(channels * 2); u16(16)
        d.append("data".data(using: .ascii)!); u32(bytes)
        return d
    }
}
