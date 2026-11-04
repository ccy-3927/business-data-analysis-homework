"""迁移趋势/季节分解与预报，明确区分历史观察、留出验证和模型预测。"""

from pathlib import Path
import sys, json, zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

c = Case(
    __file__,
    "效应分解与预测",
    "仅用教材历史数据拟合。预测值单列并明确标为模型输出，绝不写回原始数据。",
)
air = csv(c.source(find("AirPassengers.csv")))
y = pd.Series(
    air["AIR"].to_numpy(dtype=float),
    index=pd.DatetimeIndex(pd.to_datetime(air["DATE"]), freq="MS"),
    name="旅客数",
)
assert len(y) == 144 and y.index.is_unique and y.notna().all()
np.testing.assert_allclose(np.log(y), air["ln_AIR"], atol=1e-8)
profile(c, air, "航空数据")


def ets(series, horizon):
    # 在对数空间使用加法趋势/季节效应，恢复尺度后为乘法效应。
    model = ExponentialSmoothing(
        np.log(series),
        trend="add",
        seasonal="add",
        seasonal_periods=12,
        initialization_method="estimated",
    ).fit(optimized=True)
    return np.exp(model.forecast(horizon)), model


decomp = seasonal_decompose(np.log(y), model="additive", period=12)
parts = pd.DataFrame(
    {
        "对数原值": np.log(y),
        "趋势": decomp.trend,
        "季节": decomp.seasonal,
        "残差": decomp.resid,
    }
)
valid = parts.dropna()
np.testing.assert_allclose(
    valid["对数原值"], valid[["趋势", "季节", "残差"]].sum(axis=1)
)
c.table(parts, "航空对数效应分解")
fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
for ax, col in zip(axes, parts):
    ax.plot(parts.index, parts[col], color=COLORS[0])
    ax.set_ylabel(col)
    ax.grid(axis="y", alpha=0.7)
axes[0].set_title("航空旅客数的趋势与季节效应")
axes[-1].set_xlabel("月份")
c.figure(fig, "航空效应分解")
# 末12个月全留出，不用未来数据估计季节项或选择参数。
train, test = y.iloc[:-12], y.iloc[-12:]
pred, model = ets(train, 12)
naive = pd.Series(train.iloc[-12:].to_numpy(), index=test.index)
backtest = pd.DataFrame(
    {"真实观察": test, "对数HoltWinters预测": pred, "去年同月基线": naive}
)
c.table(backtest, "航空最后12个月回测")
metrics = []
for col in backtest.columns[1:]:
    err = backtest[col] - test
    metrics.append(
        {
            "模型": col,
            "MAE": err.abs().mean(),
            "RMSE": np.sqrt((err**2).mean()),
            "MAPE": (err.abs() / test).mean(),
        }
    )
c.table(pd.DataFrame(metrics), "航空回测误差")
c.line(backtest, "航空留出12个月回测", "旅客数（教材口径）", True)
forecast, fullmodel = ets(y, 24)
c.table(pd.DataFrame({"模型预测_非观测": forecast}), "航空未来24个月预测")
c.table(pd.Series({k: str(v) for k, v in fullmodel.params.items()}), "航空模型参数")
show = pd.DataFrame({"历史观察": y, "未来模型预测": forecast})
c.line(show, "航空历史与未来24个月预测", "旅客数（教材口径）", True)

for pattern, label in [
    ("信用卡消费额_含节日.xlsx", "信用卡含节日"),
    ("信用卡消费额_去除节日.xlsx", "信用卡去除节日"),
]:
    a = excel(c.source(find(pattern))).iloc[:, :2].dropna()
    a.columns = ["日期", "金额"]
    a["日期"] = pd.to_datetime(a["日期"])
    a = a.sort_values("日期").set_index("日期")
    assert a.index.is_unique and len(a) == 31

    # 一个月数据只拟合线性趋势+星期效应，不臆造节假日指示或“去节日”修正。
    def design(idx, origin):
        time = (idx - origin).days.to_numpy(dtype=float)
        return np.column_stack(
            [np.ones(len(idx)), time]
            + [(idx.dayofweek == i).astype(float) for i in range(1, 7)]
        )

    origin = a.index.min()
    X = design(a.index, origin)
    observed = a["金额"].to_numpy(dtype=float)
    beta = np.linalg.lstsq(X, observed, rcond=None)[0]
    fitted = X @ beta
    part = pd.DataFrame(
        {
            "原值": observed,
            "截距与趋势": X[:, :2] @ beta[:2],
            "星期效应": X[:, 2:] @ beta[2:],
            "拟合值": fitted,
            "残差": observed - fitted,
        },
        index=a.index,
    )
    np.testing.assert_allclose(
        part["原值"], part["截距与趋势"] + part["星期效应"] + part["残差"]
    )
    c.table(part, label + "_分解")
    c.line(
        part[["原值", "拟合值"]],
        label + "原值与趋势星期拟合",
        "金额（各自源表单位）",
        True,
    )
    b = np.linalg.lstsq(X[:-7], observed[:-7], rcond=None)[0]
    check = pd.DataFrame(
        {
            "真实观察": observed[-7:],
            "趋势星期预测": X[-7:] @ b,
            "上周同日基线": observed[-14:-7],
        },
        index=a.index[-7:],
    )
    c.table(check, label + "_最后7日回测")
    c.table(
        pd.Series(
            {
                col: np.mean(np.abs(check[col] - check["真实观察"]))
                for col in check.columns[1:]
            },
            name="MAE",
        ),
        label + "_回测误差",
    )
    future = pd.date_range(a.index[-1] + pd.Timedelta(days=1), periods=7)
    c.table(
        pd.DataFrame({"模型预测_非观测": design(future, origin) @ beta}, index=future),
        label + "_未来7日预测",
    )
    c.table(
        pd.Series(
            beta,
            index=["截距", "每日趋势", "周二", "周三", "周四", "周五", "周六", "周日"],
        ),
        label + "_模型系数_周一为基准",
    )

# WPS已保存的预测只作为参考输出，不参与模型训练；未知内部参数无法保证数值等同。
for pattern, label, sheet in [
    ("AIR_WPS自带AI功能做时间序列预测.xlsx", "航空WPS参考", "Sheet3"),
    ("信用卡消费额_WPS自带AI功能做时间序列预测.xlsx", "信用卡WPS参考", "Sheet2"),
]:
    ref = excel(c.source(find(pattern)), sheet).iloc[:, :3]
    ref.columns = ["日期", "历史值", "WPS预测值"]
    ref = ref[pd.to_datetime(ref["日期"], errors="coerce").notna()].copy()
    ref["日期"] = pd.to_datetime(ref["日期"])
    c.table(ref, label + "_教材缓存值")
    c.line(
        ref.set_index("日期")[["历史值", "WPS预测值"]],
        label + "历史与已保存预测",
        "源表单位",
    )
pbix = c.source(find("效应分解法.pbix"))
layout = json.loads(zipfile.ZipFile(pbix).read("Report/Layout").decode("utf-16-le"))
(c.out / "PowerBI预测配置.json").write_text(
    json.dumps(
        [
            json.loads(v["config"]).get("singleVisual", {})
            for s in layout["sections"]
            for v in s["visualContainers"]
        ],
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
c.notes += [
    "航空序列为144个月；分解用12个月季节周期，移动平均边缘缺失保留。预测用对数Holt-Winters，输出的是对数预测反变换后的点预测，未声称为无偏均值。末12个月回测独立于全样本最终拟合。",
    "Power BI保存了24期、季节周期12、95%置信水平等配置，但未提供内部拟合参数。Python为同类方法迁移，非Power BI算法逐数值克隆；不伪造其95%区间。WPS预测单独保留教材缓存值，不拿预测当历史训练。",
    "含节日与去除节日文件数量级不同，缺少转换说明，分别分析，不能相减认定节日效应。31天样本很短，星期拟合与未来7日输出仅供复现实操，异常日保留，没有臆造节日变量。",
]
c.done()
