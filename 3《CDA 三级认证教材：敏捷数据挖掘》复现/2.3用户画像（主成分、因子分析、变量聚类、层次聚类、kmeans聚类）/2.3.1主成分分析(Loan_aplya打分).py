"""教材案例迁移版。输入取自同名教材目录，结果见同级输出。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

c = start(
    __file__,
    "贷款申请主成分",
    "对教材10条贷款申请的五项评级标准化，提取5个主成分并保留第一分量。",
)
d = data(c, "Loan_aply.csv")
x = d.loc[:, "X1":]
z = StandardScaler().fit_transform(x)
m = PCA().fit(z)
pca_tables(c, x, m)
c.table(
    pd.DataFrame(
        m.components_, columns=x.columns, index=["PC" + str(i + 1) for i in range(5)]
    ),
    "主成分权重",
)
scores = m.transform(z)
np.testing.assert_allclose(scores[0, 0], z[0] @ m.components_[0])
c.table(d.assign(PCAScore=scores[:, 0]), "申请人主成分得分")
c.bar(pd.Series(scores[:, 0], index=d.ID), "贷款申请第一主成分", "标准化综合得分")
c.notes.append(
    f"第一分量解释率{m.explained_variance_ratio_[0]:.2%}，分数为数据压缩结果，不是已验证违约概率。"
)
c.done()
