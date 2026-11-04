"""教材案例迁移版。输入取自同名教材目录，结果见同级输出。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools, warnings

c = start(
    __file__,
    "AR MA ARMA ARIMA教学序列",
    "使用教材已提供ts_simu200.csv，不重新生成模拟序列；保留原t序号，避免编造日历日期。",
)
d = data(c, "ts_simu200.csv").set_index("t")
results = []
for col, order in [
    ("AR1_a", (1, 0, 0)),
    ("MA1_a", (0, 0, 1)),
    ("ARMA_11_b", (1, 0, 1)),
    ("ARIMA_110", (1, 1, 0)),
]:
    s = d[col]
    model = ARIMA(s.to_numpy(), order=order).fit()
    pred = pd.Series(model.forecast(6), index=np.arange(201, 207), name="预测")
    c.table(pred, col + "_预测")
    rows = []
    if col in ["ARMA_11_b", "ARIMA_110"]:
        for candidate in itertools.product(range(2), repeat=3):
            with warnings.catch_warnings(record=True):
                fit = ARIMA(s.to_numpy(), order=candidate).fit()
            rows.append(
                {
                    "order": str(candidate),
                    "AIC": fit.aic,
                    "收敛": fit.mle_retvals["converged"],
                }
            )
        c.table(pd.DataFrame(rows), col + "_阶数比较")
    results.append(
        {
            "序列": col,
            "教材order": str(order),
            "ADF原序列p": adfuller(s)[1],
            "ADF一阶差分p": adfuller(s.diff().dropna())[1],
            "残差DW": durbin_watson(model.resid),
            "残差正态检验p": stats.normaltest(model.resid).pvalue,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(s.index, s)
    axes[0, 0].set(title=col, xlabel="教材t")
    plot_acf(s, lags=20, ax=axes[0, 1])
    plot_pacf(s, lags=20, ax=axes[1, 0])
    plot_acf(model.resid, lags=20, ax=axes[1, 1])
    axes[1, 1].set_title("残差ACF")
    c.figure(fig, col + "_序列诊断")
    forecast_plot(c, s, pred, col + "_后续六期")
c.table(pd.DataFrame(results), "平稳性与残差检验")
c.notes.append(
    "这些是教材给定的模拟教学数据，并非真实业务观测；预测为模型结果。修正ARIMA阶数的keyword调用及原代码未定义param_seasonal变量。"
)
c.done()
