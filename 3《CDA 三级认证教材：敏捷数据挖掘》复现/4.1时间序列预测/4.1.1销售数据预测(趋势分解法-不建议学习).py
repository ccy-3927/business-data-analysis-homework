"""教材案例迁移版。输入取自同名教材目录，结果见同级输出。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

from prophet import Prophet

c = start(
    __file__,
    "航空客流趋势分解预测",
    "fbprophet迁移为prophet，复现加法和乘法季节模型，补充按时间留后验证。",
)
d = data(c, "air.csv")
d = pd.DataFrame({"ds": pd.to_datetime(d.DATE), "y": d.AIR}).sort_values("ds")
tr = d.iloc[:-24]
te = d.iloc[-24:]
rows = []
for mode in ["additive", "multiplicative"]:
    m = Prophet(
        growth="linear",
        seasonality_mode=mode,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.95,
        uncertainty_samples=0,
    ).fit(tr, seed=42)
    p = m.predict(te[["ds"]])
    rows.append(
        {
            "模型": mode,
            "留后24月MAE": mean_absolute_error(te.y, p.yhat),
            "留后24月RMSE": np.sqrt(mean_squared_error(te.y, p.yhat)),
        }
    )
    c.table(
        pd.DataFrame(
            {"日期": te.ds, "真实": te.y.to_numpy(), "预测": p.yhat.to_numpy()}
        ),
        mode + "_留后检验",
    )
    final = Prophet(
        growth="linear",
        seasonality_mode=mode,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.95,
    ).fit(d, seed=42)
    np.random.seed(42)  # 仅模型预测区间抽样，不生成观测数据。
    future = final.make_future_dataframe(periods=36, freq="MS")
    forecast = final.predict(future)
    c.table(
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "trend", "yearly"]],
        mode + "_预测与分解",
    )
    observed = d.set_index("ds").y
    f = forecast.set_index("ds")
    forecast_plot(c, observed, f.yhat, mode + "_客流预测", f.yhat_lower, f.yhat_upper)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    axes[0].plot(f.index, f.trend)
    axes[0].set(title=mode + "趋势", ylabel="客流趋势")
    axes[1].plot(f.index, f.yearly)
    axes[1].set(
        title="年季节项", ylabel="相对比例" if mode == "multiplicative" else "客流增量"
    )
    c.figure(fig, mode + "_趋势季节分解")
c.table(pd.DataFrame(rows), "时间留后验证")
c.notes += [
    "沿用教材航空客流数据，文件名虽写销售预测，实际目标为AIR。未将中国节日强加到来源未明确为中国的历史航空月数据；月首时间戳也不适合直接表示日级节日冲击。",
    "未来36个月是模型推算，95%区间是模型不确定性，不是新增真实客流。",
]
c.done()
