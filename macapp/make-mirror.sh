#!/bin/zsh
# Builds ScribaMemoMirror.app and, with --install, the agent that runs it.
#
# A bundle rather than a bare binary because Full Disk Access is granted by
# dragging an application into a list, and because the permission then belongs to
# this one thing. An agent rather than scriba starting it, because macOS charges
# an access to the process that asked: started from a terminal it would be the
# terminal's permission being tested, and the whole point is that no terminal and
# no Python needs one.
set -euo pipefail

cd "$(dirname "$0")"
APP="ScribaMemoMirror.app"
LABEL="dev.nerelli.scriba.memomirror"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

echo "==> building"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"
swiftc -O -parse-as-library Tools/memo-mirror.swift -o "$APP/Contents/MacOS/ScribaMemoMirror"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>Scriba Memo Mirror</string>
    <key>CFBundleIdentifier</key><string>$LABEL</string>
    <key>CFBundleExecutable</key><string>ScribaMemoMirror</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>1.0</string>
    <key>LSBackgroundOnly</key><true/>
</dict>
PLIST
echo '</plist>' >> "$APP/Contents/Info.plist"
printf 'APPL????' > "$APP/Contents/PkgInfo"
codesign --force --sign - "$APP" 2>/dev/null || true

BIN="$(pwd)/$APP/Contents/MacOS/ScribaMemoMirror"
echo "==> built $(pwd)/$APP"

if [[ "${1:-}" != "--install" ]]; then
  echo
  echo "Next, in this order:"
  echo "  1. System Settings > Privacy & Security > Full Disk Access, add:"
  echo "     $(pwd)/$APP"
  echo "  2. ./make-mirror.sh --install"
  echo
  echo "Nothing else in scriba needs that permission. This copies audio one way"
  echo "into ~/.scriba/inbox and has no other capability."
  exit 0
fi

echo "==> installing the agent"
mkdir -p "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$LABEL</string>
    <key>ProgramArguments</key>
    <array><string>$BIN</string></array>
    <key>StartInterval</key><integer>300</integer>
    <key>RunAtLoad</key><true/>
    <key>StandardOutPath</key><string>$HOME/.scriba/inbox-mirror.log</string>
    <key>StandardErrorPath</key><string>$HOME/.scriba/inbox-mirror.log</string>
</dict>
</plist>
PLIST

launchctl bootout "gui/$UID/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$UID" "$PLIST"
echo "==> running every five minutes. Log: ~/.scriba/inbox-mirror.log"
echo "    to stop:  launchctl bootout gui/$UID/$LABEL && rm $PLIST"
