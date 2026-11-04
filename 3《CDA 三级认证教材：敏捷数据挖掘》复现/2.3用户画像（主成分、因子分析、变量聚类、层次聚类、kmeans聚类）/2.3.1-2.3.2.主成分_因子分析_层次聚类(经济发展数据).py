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
    "经济发展降维与层次聚类",
    "复现标准化、PCA、稀疏PCA、因子旋转及Ward层次聚类。",
)
cities(c)
c.done()
