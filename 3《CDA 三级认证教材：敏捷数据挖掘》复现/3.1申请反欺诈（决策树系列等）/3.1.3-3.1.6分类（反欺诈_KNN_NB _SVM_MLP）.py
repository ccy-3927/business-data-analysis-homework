"""教材案例迁移版。输入取自同名教材目录，结果见同级输出。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

c = start(
    __file__,
    "申请反欺诈KNN贝叶斯SVM神经网络",
    "四类模型在同一真实数据拆分上比较；标准化封装在CV每折内。",
)
d = data(c, "loan_net.csv")
x = d.iloc[:, :4]
y = d.flag
a, b, ya, yb = train_test_split(x, y, test_size=0.4, random_state=1234, stratify=y)
specs = [
    ("KNN", KNeighborsClassifier(), {"model__n_neighbors": [1, 3, 5, 9, 17]}),
    ("朴素贝叶斯", GaussianNB(), {"model__var_smoothing": [1e-9, 1e-7]}),
    (
        "SVM",
        SVC(probability=True, random_state=42),
        {
            "model__C": [0.1, 1.0],
            "model__kernel": ["linear", "rbf"],
            "model__gamma": ["scale"],
        },
    ),
    (
        "MLP",
        MLPClassifier(max_iter=1000, random_state=42, early_stopping=True),
        {
            "model__hidden_layer_sizes": [(10,), (15,), (5, 5)],
            "model__alpha": [0.01, 1.0],
            "model__activation": ["relu", "logistic"],
        },
    ),
]
probs = {}
for name, est, grid in specs:
    p = Pipeline([("scale", StandardScaler()), ("model", est)])
    m = search(c, p, grid, a, ya, name, cv=4)
    probs[name] = m.predict_proba(b)[:, 1]
evaluate(c, yb, probs, "四类模型测试比较")
c.notes.append(
    "固定初始化种子，保留教材模型类别，候选参数见搜索结果表。SVM概率校准仅用训练数据；MLP内部训练验证用于早停，不使用最终测试集。"
)
c.done()
