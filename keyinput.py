#!/usr/bin/env python3
import sys
import os
import atexit

if os.name == 'nt':
    import msvcrt

    def check_key():
        """Windows: 非ブロッキングでキー入力を取得"""
        if msvcrt.kbhit():
            return msvcrt.getwch()
        return None

else:
    import select
    import tty
    import termios

    # ファイルディスクリプタと元の端末設定を保存
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    # 一度だけ cbreak モードに切り替え
    tty.setcbreak(fd)

    def restore_terminal():
        """終了時に元の端末設定へ戻す"""
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    # プログラム終了時（例: Ctrl+C）に復元
    atexit.register(restore_terminal)

    def check_key():
        """Linux: タイムアウト付きでキー入力を検出"""
        dr, _, _ = select.select([sys.stdin], [], [], 0.05)
        if dr:
            return sys.stdin.read(1)
        return None


if __name__ == "__main__":
    print("Press keys (Esc to exit):")
    try:
        while True:
            ch = check_key()
            if ch:
                print(f"Pressed: {repr(ch)}")
                if ord(ch) == 27:  # Escキーで終了
                    break
    except KeyboardInterrupt:
        pass
