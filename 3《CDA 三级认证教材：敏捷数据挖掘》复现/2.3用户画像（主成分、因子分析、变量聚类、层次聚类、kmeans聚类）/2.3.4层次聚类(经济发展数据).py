"""教材案例迁移版。输入取自同名教材目录，结果见同级输出。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

from portrait_common import cities

c = start(
    __file__,
    "经济发展层次聚类",
    "独立复现原始指标标准化、旋转得分、Ward聚类和两群原指标画像。",
)
cities(c, full=False)
c.done()
