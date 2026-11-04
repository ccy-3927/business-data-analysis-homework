"""教材案例迁移版。输入取自同名教材目录，结果见同级输出。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

from rules_common import association

c = start(
    __file__, "银行产品关联规则", "复现Apriori与FP-growth，并统一客户级支持度口径。"
)
association(c, "bank_prod_order.csv", 0.02)
c.done()
