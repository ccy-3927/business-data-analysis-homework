"""教材案例Python复现；结果保存在同级输出目录。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

from behavior_features import PaymentVar, DelqVar, UseVar, AccountInfo
from scorecard_common import scorecard

c = start(
    __file__,
    "贷中行为评分卡B卡",
    "以客户为单位聚合账户近3、6、12期的还款、逾期、使用率、趋势和稳定性特征。",
)
bal = data(c, "Data/CreditBalance.csv")
lab = data(c, "Data/label.csv")
assert not lab.Account_ID.duplicated().any()
joined = bal.merge(
    lab[["Cust_ID", "Account_ID", "Credit_Limit"]],
    on=["Cust_ID", "Account_ID"],
    validate="many_to_one",
)
parts = []
for months in [3, 6, 12]:
    window = joined[joined.Period.between(13 - months, 12)]
    for fn in [PaymentVar, DelqVar, UseVar]:
        parts.append(pd.DataFrame(fn(window)).add_suffix(f"_L{months}M"))
d = pd.concat(parts + [pd.DataFrame(AccountInfo(lab))], axis=1).replace(
    [np.inf, -np.inf], np.nan
)
assert not d.columns.duplicated().any()
assert d.index.is_unique
c.table(d, "客户行为宽表")
c.bar(d.Target.value_counts().sort_index(), "客户违约分布", "客户数")
scorecard(c, d, "Target", 1234, 0.6, False)
c.notes.append(
    f"原始余额表{len(bal)}条、账户表{len(lab)}条，生成{len(d)}位客户、{d.shape[1]-1}个特征。修复原循环重复累积并反复加后缀的问题；消费CV改用消费额而非余额。分区按客户完成，避免同客户跨集合。原数据未给目标观察日期，不能据此确认前瞻预测时间窗。"
)
c.done()
