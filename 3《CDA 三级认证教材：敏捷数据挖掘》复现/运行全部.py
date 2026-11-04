"""按教材目录顺序运行29个实操；也可传入文件名片段只运行匹配案例。"""

from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
scripts = sorted(root.glob("[2-7]*/**/*.py"))
if len(sys.argv) > 1:
    scripts = [p for p in scripts if any(word in p.name for word in sys.argv[1:])]
if not scripts:
    raise SystemExit("未找到匹配案例。请传入脚本文件名中的文字。")
for i, path in enumerate(scripts, 1):
    print(f"[{i}/{len(scripts)}] {path.relative_to(root)}", flush=True)
    subprocess.run([sys.executable, "-X", "utf8", str(path)], check=True)
print(f"完成 {len(scripts)} 个案例。")
