# ✋ jd-AI-handgesture — Hand Gesture Control

Control your laptop screen using hand gestures via webcam! Built with MediaPipe and OpenCV, supporting Windows system control (mouse, clicks, scroll, volume, media, dragging) and a browser-based interactive demo.

## 🌟 Features

- **☝️ 1 Finger (Index):** Move cursor
- **✌️ 2 Fingers (Index + Middle):** Left Click
- **🤟 3 Fingers:** Right Click
- **🖐️ 4 Fingers:** Scroll up/down
- **✋ Open Palm (5 Fingers):** Idle / Pause
- **✊ Fist (0 Fingers):** Click & Drag
- **🤏 Pinch (Thumb + Index):** Double Click
- **👍 Thumbs Up:** Volume Up
- **👎 Thumbs Down:** Volume Down
- **✌️ Peace / V-Sign:** Play / Pause Media
- **🤘 Rock Sign:** Mute / Unmute Toggle
- **← / → Swipes:** Alt+Tab window switch / Media Next track

---

## 🚀 Quick Start (Windows)

1. Clone the repository:
   ```bash
   git clone https://github.com/Japhet2004/jd-AI-handgesture.git
   cd jd-AI-handgesture
   ```

2. Run the application:
   - Double-click **`launch.bat`**, OR
   - Run via terminal:
     ```bash
     python -m venv venv
     venv\Scripts\pip install -r requirements.txt
     venv\Scripts\python hand_gesture_control.py
     ```

### ⌨️ Keyboard Shortcuts (Camera Window)
- **`Q` / `Esc`**: Quit
- **`H`**: Hide/show camera window
- **`O`**: Toggle floating skeleton overlay
- **`F`**: Fullscreen camera
- **`S`**: Take screenshot (`Win+Shift+S`)

---

## 🌐 Web Browser Demo

Test gesture detection directly inside your browser:
1. Open **`demo.html`** in Chrome / Edge, OR run a local server:
   ```bash
   python -m http.server 8080
   ```
2. Navigate to `http://localhost:8080/demo.html`
3. Click **"Start Camera"** and test hand gestures on the virtual screen!

---

## 📂 Project Structure

- `hand_gesture_control.py` — Main entry point for Windows app
- `gesture_detector.py` — MediaPipe hand tracking, finger counting & debouncing logic
- `system_actions.py` — Windows OS actions (mouse movement, clicks, volume control, hotkeys)
- `ui_overlay.py` — Heads-up display (HUD) & floating transparent skeleton overlay
- `demo.html` — Interactive Web/HTML5 version using MediaPipe JS
- `config.json` — Customizable settings (camera index, smoothing, scroll speed)
- `launch.bat` — One-click Windows starter script
- `requirements.txt` — Python dependencies
