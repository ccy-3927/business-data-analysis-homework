"""一键运行第二章全部案例；失败立即停止，不生成审阅页面。"""

from pathlib import Path
import subprocess, sys, time, json

base = Path(__file__).resolve().parent
records = []
for script in sorted(base.glob("0[1-7]_*.py")):
    start = time.perf_counter()
    subprocess.run([sys.executable, "-X", "utf8", str(script)], check=True)
    records.append(
        {
            "脚本": script.name,
            "状态": "成功",
            "用时秒": round(time.perf_counter() - start, 2),
        }
    )
(base / "运行结果.json").write_text(
    json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("第二章全部运行成功；图表、结果及说明见输出目录。")
