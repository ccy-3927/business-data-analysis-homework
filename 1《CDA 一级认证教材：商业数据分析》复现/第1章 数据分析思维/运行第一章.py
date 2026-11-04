"""在任意工作目录运行：python 运行第一章.py。失败时立即停止。"""

from pathlib import Path
import subprocess, sys

base = Path(__file__).resolve().parent
for script in sorted(base.glob("0[1-3]_*.py")):
    subprocess.run([sys.executable, "-X", "utf8", str(script)], check=True)
print("第一章全部运行完成。结果见本章输出目录。")
