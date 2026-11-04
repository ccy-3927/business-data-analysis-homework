"""教材案例迁移版。输入取自同名教材目录，结果见同级输出。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

import itertools, warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

c = start(
    __file__,
    "拖拉机销售SARIMA",
    "复现对数变换、季节分解、64组阶数AIC搜索、残差诊断和36期预测。",
)
d = data(c, "tractor_sales.csv")
ts = pd.Series(
    d.Number_of_Sold.to_numpy(),
    index=pd.DatetimeIndex(pd.to_datetime(d.Year_Month, format="%Y-%b"), freq="MS"),
    name="销量",
)
train = ts.iloc[:-24]
test = ts.iloc[-24:]
fig = seasonal_decompose(ts, model="multiplicative", period=12).plot()
fig.set_size_inches(12, 9)
c.figure(fig, "销量季节分解")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
diff = np.log(train).diff().dropna()
plot_acf(diff, lags=30, ax=axes[0])
plot_pacf(diff, lags=30, ax=axes[1])
c.figure(fig, "训练序列差分ACF与PACF")
rows = []
candidates = []
for order in itertools.product(range(2), repeat=3):
    for season in itertools.product(range(2), repeat=3):
        with warnings.catch_warnings(record=True) as ws:
            try:
                m = SARIMAX(
                    np.log(train),
                    order=order,
                    seasonal_order=(*season, 12),
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                ).fit(disp=False, maxiter=150)
                converged = m.mle_retvals.get("converged", False)
                rows.append(
                    {
                        "order": str(order),
                        "seasonal": str(season),
                        "AIC": m.aic,
                        "收敛": converged,
                        "警告": "; ".join(str(w.message) for w in ws),
                    }
                )
                if converged and np.isfinite(m.aic):
                    candidates.append((m.aic, order, season, m))
            except (ValueError, np.linalg.LinAlgError) as e:
                rows.append(
                    {
                        "order": str(order),
                        "seasonal": str(season),
                        "收敛": False,
                        "警告": str(e),
                    }
                )
assert candidates
_, order, season, m = min(candidates, key=lambda x: x[0])
c.table(pd.DataFrame(rows), "训练AIC搜索")
pred = np.exp(m.forecast(24))
c.table(pd.DataFrame({"真实": test, "预测中位数": pred}), "最后24月验证")
c.table(
    pd.Series(
        {
            "MAE": mean_absolute_error(test, pred),
            "RMSE": np.sqrt(mean_squared_error(test, pred)),
        }
    ),
    "留后验证指标",
)
with warnings.catch_warnings(record=True) as ws:
    final = SARIMAX(np.log(ts), order=order, seasonal_order=(*season, 12)).fit(
        disp=False, maxiter=300
    )
assert final.mle_retvals.get("converged", False)
fc = final.get_forecast(36)
ci = np.exp(fc.conf_int())
median = np.exp(fc.predicted_mean)
assert median.index[0] > ts.index[-1]
c.table(
    pd.DataFrame(
        {"预测中位数": median, "95%下界": ci.iloc[:, 0], "95%上界": ci.iloc[:, 1]}
    ),
    "未来36月预测",
)
forecast_plot(c, ts, median, "拖拉机未来36月", ci.iloc[:, 0], ci.iloc[:, 1])
fig = final.plot_diagnostics(lags=24, figsize=(12, 9))
c.figure(fig, "最终模型残差诊断")
c.notes += [
    f"训练期搜索选中order={order}、seasonal={season}、周期12。最后24月只验证；之后按固定阶数全量重拟合。预测从最后观测的下个月开始，修正教材日期重复。",
    "对数预测反变换为原尺度中位数及95%模型预测区间，不误称原尺度均值。搜索未收敛候选已记录并排除。",
]
c.done()
