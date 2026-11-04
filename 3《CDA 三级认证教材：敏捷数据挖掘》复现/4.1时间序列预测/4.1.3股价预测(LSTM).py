"""教材案例迁移版。输入取自同名教材目录，结果见同级输出。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

from numpy_lstm import NumpyLSTM
from sklearn.preprocessing import MinMaxScaler

c = start(
    __file__,
    "Google股价双层LSTM",
    "迁移为NumPy显式双层32单元LSTM：60步窗口、末层Dropout0.2、Adam、100轮、批16、不打乱时间。",
)
d = data(c, "google_stock_prics.csv")
d.Date = pd.to_datetime(d.Date)
d = d.sort_values("Date").reset_index(drop=True)
values = d[["Open", "Close"]].to_numpy()
cutoff = round(0.8 * len(d))
split = int(0.8 * cutoff)
scaler = MinMaxScaler().fit(values[:split])
scaled = scaler.transform(values)
ids = np.arange(60, len(d))
x = np.stack([scaled[i - 60 : i] for i in ids])
y = scaled[ids]
trainmask = ids < split
valmask = (ids >= split) & (ids < cutoff)
testmask = ids >= cutoff
xt, yt = x[trainmask], y[trainmask]
xv, yv = x[valmask], y[valmask]
xe = x[testmask]
model = NumpyLSTM()
gradient_error = model.check_gradient(xt[:2, :3], yt[:2])
history = []
best = float("inf")
best_state = None
with threadpool_limits(1):
    for epoch in range(100):
        total = 0
        for start in range(0, len(xt), 16):
            xx, yy = xt[start : start + 16], yt[start : start + 16]
            loss, grad = model.loss_grad(xx, yy, training=True)
            model.update(grad)
            total += loss * len(xx)
        vl = np.mean((model.predict(xv) - yv) ** 2)
        history.append(
            {"轮次": epoch + 1, "训练MSE": total / len(xt), "内部验证MSE": vl}
        )
        if vl < best:
            best = vl
            best_state = {k: v.copy() for k, v in model.params.items()}
            best_epoch = epoch + 1
        if (epoch + 1) % 10 == 0:
            print("LSTM epoch", epoch + 1, "validation", vl, flush=True)
    model.params = best_state
    prediction = scaler.inverse_transform(model.predict(xe))
testids = ids[testmask]
actual = values[testids]
naive = values[testids - 1]
rows = []
out = pd.DataFrame({"日期": d.Date.iloc[testids].to_numpy()})
for j, name in enumerate(["Open", "Close"]):
    out[name + "_真实"] = actual[:, j]
    out[name + "_LSTM"] = prediction[:, j]
    out[name + "_昨日基线"] = naive[:, j]
    for label, p in [("LSTM", prediction), ("昨日值基线", naive)]:
        rows.append(
            {
                "目标": name,
                "模型": label,
                "MAE": mean_absolute_error(actual[:, j], p[:, j]),
                "RMSE": np.sqrt(mean_squared_error(actual[:, j], p[:, j])),
            }
        )
c.table(out, "时间测试预测")
c.table(pd.DataFrame(rows), "同目标同日期评估")
c.table(pd.DataFrame(history), "训练损失")
c.line(pd.DataFrame(history).set_index("轮次"), "训练与内部验证损失", "MSE")
c.line(
    out.set_index("日期")[["Close_真实", "Close_LSTM", "Close_昨日基线"]],
    "收盘价一步预测",
    "价格",
    interactive=True,
)
np.savez_compressed(c.out / "LSTM权重.npz", **model.params)
joblib.dump(scaler, c.out / "价格归一化.joblib")
c.notes += [
    f"使用真实训练窗口数值差分核对BPTT，最大梯度绝对误差{gradient_error:.3g}。内部验证最优轮次{best_epoch}；最终测试{len(testids)}个交易日。",
    "验证期为前80%时间段内最后20%，归一化只拟合其之前训练段；最终测试每次使用前60天实际观测做滚动一步预测。修正教材Open/Close错列和基线日期不对齐。",
    "PyTorch下载失败后采用NumPy显式LSTM，不以其他回归模型冒充LSTM。初始化与旧Keras不同，加入梯度范数上限5用于稳定优化；预测效果如实与昨日值基线比较。",
]
c.done()
