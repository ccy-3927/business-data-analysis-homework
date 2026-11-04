"""教材案例迁移版。输入取自同名教材目录，结果见同级输出。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

from scipy.optimize import milp, Bounds, LinearConstraint
from optimization_common import read_linear

c = start(
    __file__,
    "营销资源整数分配",
    "读取campaign.lp的27个变量、收益系数与资源约束，用HiGHS整数求解。",
)
obj, A, b, lo, hi, names = read_linear(c, "campaign.lp")
r = milp(
    -obj,
    integrality=np.ones(len(obj)),
    bounds=Bounds(lo, hi),
    constraints=LinearConstraint(A, -np.inf, b),
)
assert r.success
np.testing.assert_allclose(r.x, np.rint(r.x), atol=1e-7)
assert np.all(A @ r.x <= b + 1e-7)
allocation = np.rint(r.x).astype(int).reshape(3, 9).T
table = pd.DataFrame(
    allocation,
    index=["客户" + str(i + 1) for i in range(9)],
    columns=["方案1", "方案2", "方案3"],
)
c.table(table, "最优营销分配")
heat(c, table, "客户与营销方案分配")
c.table(
    pd.DataFrame(
        {"变量": names, "收益系数": obj, "是否选择": np.rint(r.x).astype(int)}
    ),
    "分配收益明细",
)
c.notes.append(
    f"最优总收益{obj@r.x:g}。每客户至多一种方案，每方案至多分配3名客户。原LP的Generals遗漏x19；按0/1分配语义统一设为整数，并核对所有原约束。"
)
c.done()
