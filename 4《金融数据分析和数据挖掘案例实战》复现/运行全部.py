"""运行全部案例，或传入文件名关键词运行指定案例。"""

from pathlib import Path
import os, subprocess, sys

root = Path(__file__).resolve().parent
files = sorted([*root.glob("第*/**/*.py"), *root.glob("补充*/*.py")])
if len(sys.argv) > 1:
    files = [p for p in files if any(word in p.name for word in sys.argv[1:])]
if not files:
    raise SystemExit("未匹配到案例文件名")
env = os.environ.copy()
env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
for p in files:
    print("运行：" + str(p.relative_to(root)), flush=True)
    subprocess.run([sys.executable, str(p)], check=True, env=env)
