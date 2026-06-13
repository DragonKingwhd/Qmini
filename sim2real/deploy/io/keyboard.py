"""Keyboard velocity command source for run_qmini (--keyboard).

键位:
    w / s : vx +0.05 / -0.05   (clip 到 [0, 0.30])
    a / d : wz +0.10 / -0.10   (a=左转/逆时针为正, clip 到 ±0.50)
    空格/x: 全部归零(原地踏步)
    vy 恒为 0 —— 训练命令分布里 vy≡0,没法侧移。

训练分布(UniLab QminiJoystickFlat): vx∈[0.10,0.30], wz∈[-0.5,0.5];
vx<0.10 只有 2% standing 环境见过 → 低速段(0<vx<0.1)是分布边缘,可能发飘。

终端在第一次 read() 时才切 cbreak(免回车逐键读),所以启动阶段的 input()
确认门不受影响;close() 恢复终端。Ctrl+C 在 cbreak 下仍然有效。
"""

from __future__ import annotations

import select
import sys
import termios
import tty

import numpy as np

from .interfaces import CommandSource


class KeyboardCommand(CommandSource):
    def __init__(
        self,
        vx_step: float = 0.05,
        wz_step: float = 0.10,
        vx_max: float = 0.30,
        wz_max: float = 0.50,
    ) -> None:
        self._vx = 0.0
        self._wz = 0.0
        self._vx_step = float(vx_step)
        self._wz_step = float(wz_step)
        self._vx_max = float(vx_max)
        self._wz_max = float(wz_max)
        self._fd = sys.stdin.fileno()
        self._saved: list | None = None  # termios state; lazy cbreak

    def _ensure_cbreak(self) -> None:
        if self._saved is None:
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)

    def read(self) -> np.ndarray:
        self._ensure_cbreak()
        changed = False
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1).lower()
            if ch == "w":
                self._vx = min(self._vx + self._vx_step, self._vx_max)
            elif ch == "s":
                self._vx = max(self._vx - self._vx_step, 0.0)
            elif ch == "a":
                self._wz = min(self._wz + self._wz_step, self._wz_max)
            elif ch == "d":
                self._wz = max(self._wz - self._wz_step, -self._wz_max)
            elif ch in (" ", "x"):
                self._vx = 0.0
                self._wz = 0.0
            else:
                continue
            changed = True
        if changed:
            print(f"\r[cmd] vx={self._vx:+.2f} m/s  wz={self._wz:+.2f} rad/s   ",
                  end="", flush=True)
        return np.array([self._vx, 0.0, self._wz], dtype=np.float32)

    def close(self) -> None:
        if self._saved is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            except Exception:
                pass
            self._saved = None
            print()  # 把 \r HUD 行结束掉
