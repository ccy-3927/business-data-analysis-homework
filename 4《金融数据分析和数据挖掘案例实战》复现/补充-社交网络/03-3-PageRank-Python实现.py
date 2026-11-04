"""教材案例Python复现；结果保存在同级输出目录。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

from network_common import *

c = start(
    __file__,
    "PageRank幂迭代",
    "使用教材固定转移矩阵修复迭代不更新的错误；跳过随机生成网络。",
)
m = np.array(
    [[0, 0.5, 0, 0.5], [1 / 3, 0, 0, 1 / 3], [1 / 3, 0.5, 0, 0], [1 / 3, 0, 1, 0]]
)
c.table(pd.DataFrame({"原始列和": m.sum(axis=0)}), "教材转移矩阵核对")
m = m / m.sum(axis=0)
np.testing.assert_allclose(m.sum(axis=0), 1)
c.table(pd.DataFrame(m), "归一化转移矩阵")
v = np.ones(4) / 4
history = []
for i in range(1000):
    new = m @ v
    err = np.linalg.norm(new - v, 1)
    history.append([i + 1, *new, err])
    v = new
    if err < 1e-10:
        break
np.testing.assert_allclose(m @ v, v, atol=1e-9)
np.testing.assert_allclose(v.sum(), 1)
h = pd.DataFrame(
    history, columns=["迭代", "节点1", "节点2", "节点3", "节点4", "L1变化"]
).set_index("迭代")
c.table(h, "幂迭代过程")
c.line(h.drop(columns="L1变化"), "PageRank收敛过程", "概率")
c.bar(pd.Series(v, index=["节点1", "节点2", "节点3", "节点4"]), "稳定PageRank", "概率")
c.notes.append(
    "教材矩阵第4列之和为5/6，不能保持概率质量；按列归一化后计算无阻尼平稳分布，保留原矩阵列和核对。矩阵列表示来源节点。原MV_computer未更新v，已修正。未执行随机网络生成函数。"
)
c.done()
