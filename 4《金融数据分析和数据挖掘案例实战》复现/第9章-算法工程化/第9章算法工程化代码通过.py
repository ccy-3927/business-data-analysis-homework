"""教材案例Python复现；结果保存在同级输出目录。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

from sklearn.base import BaseEstimator, TransformerMixin

c = start(
    __file__,
    "时间特征工程化",
    "将教材时间转换器迁移为可组合的sklearn接口，使用配套银行交易日期，不制造示例业务记录。",
)


class CustomTimeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, time_col="date"):
        self.time_col = time_col

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        out = x.copy()
        t = pd.to_datetime(out[self.time_col])
        out["year"] = t.dt.year
        out["month"] = t.dt.month
        out["day"] = t.dt.day
        out["week_day"] = t.dt.dayofweek
        out["day_of_year"] = t.dt.dayofyear
        out["week_of_year"] = t.dt.isocalendar().week.to_numpy()
        return out


class CustomTimeBooleanTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, time_col="date"):
        self.time_col = time_col

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        out = x.copy()
        t = pd.to_datetime(out[self.time_col])
        for attr in [
            "is_leap_year",
            "is_month_start",
            "is_month_end",
            "is_quarter_start",
            "is_quarter_end",
            "is_year_start",
            "is_year_end",
        ]:
            out[attr] = getattr(t.dt, attr)
        out["is_weekend"] = t.dt.dayofweek >= 5
        return out


class CustomTimeSeriesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=("交易金额",)):
        self.cols = cols

    def fit(self, x, y=None):
        self.correlations_ = {
            col: {k: x[col].autocorr(k) for k in [1, 2]} for col in self.cols
        }
        return self

    def transform(self, x):
        out = x.copy()
        for col in self.cols:
            for lag in [1, 2, 3]:
                out[f"{col}_lag{lag}"] = out[col].shift(lag)
            out[col + "_前三期均值"] = out[col].shift(1).rolling(3).mean()
            out[col + "_前两期差"] = out[col].shift(1) - out[col].shift(2)
            for lag in [1, 2]:
                out[f"{col}_训练自相关{lag}"] = self.correlations_[col][lag]
        return out


p = next(SOURCE.glob("第2*/mysql_csv/Kdd99_trans.csv"))
t = csv(
    c.source(p),
    header=None,
    names=[
        "trans_id",
        "account_id",
        "date",
        "type",
        "operation",
        "amount",
        "balance",
        "symbol",
        "bank",
        "account",
    ],
    low_memory=False,
)
d = (
    t.groupby("date")
    .agg(交易金额=("amount", "sum"), 交易笔数=("trans_id", "size"))
    .reset_index()
    .sort_values("date")
)
cut = int(len(d) * 0.7)
pipe = Pipeline(
    [
        ("calendar", CustomTimeTransformer()),
        ("boolean", CustomTimeBooleanTransformer()),
        ("history", CustomTimeSeriesTransformer()),
    ]
)
pipe.fit(d.iloc[:cut])
out = pipe.transform(d)
assert out["交易金额_lag2"].equals(d["交易金额"].shift(2))
c.table(out, "真实交易日时间特征")
c.line(
    out.set_index("date")[["交易金额", "交易金额_前三期均值"]],
    "银行交易金额与历史均值",
    "交易金额",
    interactive=True,
)
c.bar(out.groupby("week_day")["交易笔数"].mean(), "各星期平均交易笔数", "平均笔数")
c.notes.append(
    "教材原两行手写时间示例不作为实证数据，改用本资料第2章交易流水演示相同转换器。日期仅精确到日，不编造时分秒或节假日。lag1/2/3修复原来全为shift(1)的问题；自相关仅由前70%训练期估计。滞后按已有交易日顺序计算，不把无记录日期当作零交易。"
)
c.done()
