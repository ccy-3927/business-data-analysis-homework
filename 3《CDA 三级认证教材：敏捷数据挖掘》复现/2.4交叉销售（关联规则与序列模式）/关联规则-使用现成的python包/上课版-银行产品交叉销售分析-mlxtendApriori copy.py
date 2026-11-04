"""教材案例迁移版。输入取自同名教材目录，结果见同级输出。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

from rules_common import association

c = start(__file__, "关联规则课堂版", "迁移失效的个人电脑绝对路径，使用同目录教材CSV。")
association(c, "bank_prod_order(1).csv", 0.01)
c.done()
