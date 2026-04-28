from pynput import keyboard

print("キーを押してください。Escで終了します。")

def on_press(key):
    print(f"押した: {key}  (内部表現: {str(key)})")

def on_release(key):
    if key == keyboard.Key.esc:
        return False

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
