"""依次运行本章实操，失败即停止。"""

from pathlib import Path
import subprocess, sys, json, time

folder = Path(__file__).resolve().parent
results = []
for path in sorted(folder.glob("0[1-9]_*.py")):
    started = time.perf_counter()
    subprocess.run([sys.executable, "-X", "utf8", str(path)], check=True)
    results.append(
        {
            "脚本": path.name,
            "状态": "成功",
            "用时秒": round(time.perf_counter() - started, 2),
        }
    )
(folder / "运行结果.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
)
