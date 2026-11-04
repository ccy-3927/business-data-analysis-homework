"""教材案例迁移版。输入取自同名教材目录，结果见同级输出。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

from scipy.optimize import linprog
from optimization_common import read_linear

c = start(
    __file__,
    "产品生产线性规划",
    "CPLEX迁移到SciPy HiGHS，直接读取productivity.lp目标与约束。",
)
obj, A, b, lo, hi, names = read_linear(c, "productivity.lp")
r = linprog(-obj, A_ub=A, b_ub=b, bounds=list(zip(lo, hi)), method="highs")
assert r.success and np.all(A @ r.x <= b + 1e-8)
c.table(pd.Series(r.x, index=names, name="最优数量"), "最优生产方案")
c.table(
    pd.DataFrame({"资源上限": b, "使用量": A @ r.x, "剩余": b - A @ r.x}), "约束核对"
)
fig, ax = plt.subplots(figsize=(9, 6))
xx = np.linspace(0, 300, 400)
cap = np.minimum.reduce([300 - xx, 400 - 2 * xx, np.full_like(xx, 250)])
ax.fill_between(
    xx,
    0,
    np.maximum(cap, 0),
    where=cap >= 0,
    alpha=0.2,
    color=COLORS[0],
    label="可行域",
)
for row, rhs in zip(A, b):
    if row[1]:
        ax.plot(xx, (rhs - row[0] * xx) / row[1], ls="--", alpha=0.6)
ax.scatter(*r.x, color=COLORS[4], s=70, label=f"最优点({r.x[0]:g}, {r.x[1]:g})")
ax.set(
    xlim=(0, 300),
    ylim=(0, 320),
    xlabel=names[0],
    ylabel=names[1],
    title="生产方案可行域与最优点",
)
ax.legend()
c.figure(fig, "线性规划可行域")
c.notes.append(
    f"最大目标值{obj@r.x:g}。Notebook开头文字的4变量例题与实际productivity.lp不同，本脚本按其实际读取的2变量生产模型求解，未混用约束。"
)
c.done()
