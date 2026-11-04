"""教材案例迁移版。输入取自同名教材目录，结果见同级输出。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

from scipy.optimize import minimize

c = start(__file__, "二次规划", "按qp1.lp的二次目标及a+b=1、非负约束迁移到SciPy。")
p = c.source(c.notebook.parent / "qp1.lp")
source = p.read_text()
assert "[2 a^2 + b^2 + a * b] /2 + a + b" in source
H = np.array([[2.0, 0.5], [0.5, 1.0]])
assert np.linalg.eigvalsh(H).min() > 0
fun = lambda x: 0.5 * x @ H @ x + x.sum()
jac = lambda x: H @ x + 1
r = minimize(
    fun,
    [0.5, 0.5],
    jac=jac,
    bounds=[(0, None), (0, None)],
    constraints={
        "type": "eq",
        "fun": lambda x: x.sum() - 1,
        "jac": lambda x: np.ones(2),
    },
    method="SLSQP",
    options={"ftol": 1e-12},
)
assert r.success and abs(r.x.sum() - 1) < 1e-9
np.testing.assert_allclose(r.x, [0.25, 0.75], atol=1e-8)
c.table(
    pd.Series({"a": r.x[0], "b": r.x[1], "目标值": r.fun, "等式误差": r.x.sum() - 1}),
    "最优解与核验",
)
a = np.linspace(0, 1, 301)
values = np.array([fun(np.array([t, 1 - t])) for t in a])
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(a, values)
ax.scatter(r.x[0], r.fun, color=COLORS[4], s=60)
ax.set(title="可行线段上的二次目标", xlabel="a（b=1−a）", ylabel="目标函数")
c.figure(fig, "二次规划最优点")
c.notes.append(
    "最优a=0.25、b=0.75，目标1.4375。Hessian正定，因此可行集上的该解为唯一全局最优；曲线网格仅计算目标函数，不是生成业务数据。"
)
c.done()
