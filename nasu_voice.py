import sys
import os
import json
import tempfile
import threading
import time
import ctypes
import tkinter as tk

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import pyperclip
from groq import Groq
from pynput import keyboard as pynput_kb

# 多重起動防止
import msvcrt
_LOCK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".nasu_voice.lock")
try:
    _lock_fd = open(_LOCK_FILE, "w")
    msvcrt.locking(_lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
except (IOError, OSError):
    sys.exit(0)

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nasu_config.json")
SAMPLE_RATE = 16000


def load_config():
    if not os.path.exists(CONFIG_FILE):
        return {}
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return json.load(f)


def save_config(config):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def key_to_str(key):
    return str(key)


def str_to_key(s):
    s = s.strip()
    if s.startswith("Key."):
        return pynput_kb.Key[s[4:]]
    char = s.strip("'\"")
    return pynput_kb.KeyCode.from_char(char)


def key_display(key):
    try:
        return key.char.upper()
    except AttributeError:
        return str(key).replace("Key.", "").upper()


class ApiKeyDialog:
    def __init__(self):
        self.result = None
        self.root = tk.Tk()
        self.root.title("NAS Voice - API キー設定")
        self.root.geometry("500x180")
        self.root.resizable(False, False)

        tk.Label(self.root, text="Groq API キーを入力してください", font=("Yu Gothic UI", 13), pady=15).pack()
        tk.Label(self.root, text="https://console.groq.com でキーを取得できます", font=("Yu Gothic UI", 10), fg="#666").pack()
        self.entry = tk.Entry(self.root, font=("Yu Gothic UI", 12), width=48, show="*")
        self.entry.pack(pady=8)
        tk.Button(self.root, text="決定", font=("Yu Gothic UI", 12), command=self._confirm, width=10).pack(pady=8)
        self.root.bind("<Return>", lambda e: self._confirm())
        self.root.mainloop()

    def _confirm(self):
        self.result = self.entry.get().strip()
        self.root.destroy()


class KeySetupDialog:
    def __init__(self):
        self.captured_key = None
        self.root = tk.Tk()
        self.root.title("NAS Voice - キー設定")
        self.root.geometry("400x200")
        self.root.resizable(False, False)

        tk.Label(self.root, text="録音に使うキーを押してください", font=("Yu Gothic UI", 14), pady=20).pack()
        self.key_label = tk.Label(self.root, text="待機中...", font=("Yu Gothic UI", 16, "bold"), fg="#aaa")
        self.key_label.pack(pady=5)
        self.btn = tk.Button(self.root, text="このキーで決定", font=("Yu Gothic UI", 12),
                             state=tk.DISABLED, command=self._confirm)
        self.btn.pack(pady=12)

        self.listener = pynput_kb.Listener(on_press=self._on_key)
        self.listener.start()
        self.root.mainloop()

    def _on_key(self, key):
        self.captured_key = key
        name = key_display(key)
        self.root.after(0, lambda: self.key_label.config(text=name, fg="#1a8a1a"))
        self.root.after(0, lambda: self.btn.config(state=tk.NORMAL))

    def _confirm(self):
        self.listener.stop()
        self.root.destroy()


class VoiceDaemon:
    def __init__(self, config, hotkey):
        self.config = config
        self.hotkey = hotkey
        self.client = Groq(api_key=config["api_key"])
        self.recording = []
        self.is_recording = False
        self.stream = None

        self.root = tk.Tk()
        self.root.withdraw()
        self.root.update()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.92)
        self.root.configure(bg="#1a1a2e")

        self.label = tk.Label(
            self.root, text="", font=("Yu Gothic UI", 13),
            bg="#1a1a2e", fg="white", padx=20, pady=10
        )
        self.label.pack()

        self._start_listener()
        print(f"NAS Voice 起動完了。{key_display(hotkey)} を押している間、録音します")

    def _start_listener(self):
        hotkey_str = key_to_str(self.hotkey)

        def on_press(key):
            if key_to_str(key) == hotkey_str and not self.is_recording:
                self.root.after(0, self._start)

        def on_release(key):
            if key_to_str(key) == hotkey_str and self.is_recording:
                self.root.after(0, self._stop)

        listener = pynput_kb.Listener(on_press=on_press, on_release=on_release)
        listener.daemon = True
        listener.start()

    def _show(self, text, color="white"):
        self.label.config(text=text, fg=color)
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        w = self.root.winfo_reqwidth()
        self.root.geometry(f"+{(sw - w) // 2}+40")
        self.root.deiconify()
        self.root.update()

    def _hide(self):
        self.root.withdraw()

    def _start(self):
        self.recording = []
        self.is_recording = True
        self._show("● 録音中...", "#ff4444")

        def callback(indata, frames, t, status):
            self.recording.append(indata.copy())

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=callback
        )
        self.stream.start()

    def _stop(self):
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self._show("変換中...", "#ffcc44")
        threading.Thread(target=self._transcribe, daemon=True).start()

    def _transcribe(self):
        if not self.recording:
            self.root.after(0, self._hide)
            return

        audio_data = np.concatenate(self.recording, axis=0)

        rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
        if rms < 200:
            self.root.after(0, self._hide)
            return

        # 末尾無音トリム（Whisperの幻覚テキスト防止）
        chunk_size = SAMPLE_RATE * 100 // 1000
        chunks = [audio_data[i:i + chunk_size] for i in range(0, len(audio_data), chunk_size)]
        while chunks and np.sqrt(np.mean(chunks[-1].astype(np.float32) ** 2)) < 200:
            chunks.pop()
        if chunks:
            audio_data = np.concatenate(chunks)

        tmp_path = tempfile.mktemp(suffix=".wav")
        wav.write(tmp_path, SAMPLE_RATE, audio_data)

        try:
            with open(tmp_path, "rb") as f:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-large-v3-turbo",
                    file=f,
                    language=self.config.get("language", "ja"),
                    prompt=self.config.get("prompt", ""),
                )
            text = transcription.text
            pyperclip.copy(text)

            preview = text[:25] + ("..." if len(text) > 25 else "")
            self.root.after(0, lambda: self._show(f"完了: {preview}", "#44dd44"))
            time.sleep(0.2)
            ctypes.windll.user32.keybd_event(0x11, 0, 0, 0)
            ctypes.windll.user32.keybd_event(0x56, 0, 0, 0)
            ctypes.windll.user32.keybd_event(0x56, 0, 2, 0)
            ctypes.windll.user32.keybd_event(0x11, 0, 2, 0)
            time.sleep(1.5)
            self.root.after(0, self._hide)

        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                self.root.after(0, lambda: self._show("本日の無料枠を使い切りました", "#ff8800"))
            else:
                self.root.after(0, lambda: self._show("エラーが発生しました", "#ff4444"))
            time.sleep(2.5)
            self.root.after(0, self._hide)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def run(self):
        self.root.mainloop()


def main():
    config = load_config()

    if not config.get("api_key"):
        dialog = ApiKeyDialog()
        if not dialog.result:
            sys.exit(0)
        config["api_key"] = dialog.result
        save_config(config)

    hotkey_str = config.get("hotkey", "")
    if not hotkey_str:
        dialog = KeySetupDialog()
        if not dialog.captured_key:
            sys.exit(0)
        config["hotkey"] = key_to_str(dialog.captured_key)
        save_config(config)
        hotkey = dialog.captured_key
    else:
        hotkey = str_to_key(hotkey_str)

    daemon = VoiceDaemon(config, hotkey)
    daemon.run()


if __name__ == "__main__":
    main()
