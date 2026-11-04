"""教材Notebook的可运行迁移版；原单元格及变化见同目录对照文档。"""

from pathlib import Path
import sys

for folder in Path(__file__).resolve().parents:
    if (folder / "analysis_common.py").exists():
        sys.path.insert(0, str(folder))
        break
from analysis_common import *

c = start(
    __file__,
    "数据整合和清洗",
    "复现读取、查询、连接、合并、排序、分组、透视、重编码、缺失及异常处理。",
)
a = data(c, "sample.csv")
big = data(c, "data/big.csv")
profile(c, a, "样本")
profile(c, big, "银行记录")
c.table(a[["id", "name"]].head(2), "指定列与前两行")
chunks = list(pd.read_csv(c.notebook.parent / "data/big.csv", chunksize=900))
pd.testing.assert_frame_equal(pd.concat(chunks, ignore_index=True), big)
c.table(pd.Series([len(x) for x in chunks], name="行数"), "分块读取检查")
c.table(a.replace(99999, np.nan), "特殊缺失值识别")
c.table(a, "CSV导出结果")
# 原cell10随机数据替换为本章已有银行数据，展示相同的行列方法。
b = big.head(4).copy()
b["柜台减ATM"] = b["CNT_TBM"] - b["CNT_ATM"]
b["柜台加ATM"] = b["CNT_TBM"] + b["CNT_ATM"]
c.table(b, "行列选择与派生列")
c.table(b.iloc[:2, :2], "iloc行列切片")
c.table(b.drop(columns=["柜台加ATM"]), "删除派生列")
q = literal(c, 22, "sample")
c.table(q, "教材固定查询示例")
queries = {
    "分数超过70": q.score > 70,
    "分数超过70且组1": (q.score > 70) & q.group.eq(1),
    "非组1": ~q.group.eq(1),
    "分数70至80": q.score.between(70, 80),
    "姓名列表": q.name.isin(["Bob", "Lindy"]),
    "姓名含M": q.name.str.contains("M"),
}
for label, mask in queries.items():
    c.table(q[mask], "条件查询_" + label)
c.table(q.query("score>90"), "query表达式")
left, right = literal(c, 34, "df1"), literal(c, 34, "df2")
for how in ["inner", "left", "right", "outer"]:
    c.table(left.merge(right, on="id", how=how), "横向连接_" + how)
left, right = literal(c, 42, "df1"), literal(c, 42, "df2")
left.index = [1, 2, 3]
right.index = [1, 3, 2]
c.table(left.join(right), "索引对齐连接")
left, right = literal(c, 48, "df1"), literal(c, 48, "df2")
union = pd.concat([left, right], ignore_index=True)
c.table(union, "纵向合并")
c.table(union.drop_duplicates(), "合并后去重")
c.table(
    pd.concat([left, left.rename(columns={"col": "new_col"})], ignore_index=True),
    "不同列名合并",
)
s = literal(c, 56, "sample")
c.table(s.sort_values("score", ascending=False, na_position="last"), "排序缺失置后")
c.table(s.sort_values(["group", "score"]), "多列排序")
s = literal(c, 60, "sample")
c.table(
    s.groupby(["grade", "class"])[["math", "chinese"]].agg(["min", "max", "mean"]),
    "多级分组汇总",
)
t = literal(c, 70, "table")
c.table(
    t.pivot_table(
        index="cust_id", columns="type", values="Monetary", aggfunc="sum", fill_value=0
    ),
    "客户类型透视",
)
s = literal(c, 74, "sample")
s["score"] = s.score.replace(999, np.nan)
s["class_n"] = s.group.map({1: "class1", 2: "class2"})
c.table(s, "异常编码与条件赋值")
s = literal(c, 82, "sample")
c.table(s[s.duplicated()], "重复记录")
c.table(s.drop_duplicates(), "整行去重")
c.table(s.drop_duplicates("id"), "按ID保留首条")
s = literal(c, 87, "sample")
c.table(s.isna().mean().rename("缺失比例"), "缺失比例")
s["均值填补"] = s.score.fillna(s.score.mean())
s["中位数填补"] = s.score.fillna(s.score.median())
s["原值缺失"] = s.score.isna()
c.table(s, "教材固定缺失示例处理")
v = big["CNT_TBM"]
lo, hi = v.quantile([0.01, 0.99])
capped = v.clip(lo, hi)
c.table(
    pd.DataFrame(
        {
            "原值": v,
            "盖帽值": capped,
            "是否调整": v.ne(capped),
            "等宽箱": pd.cut(v, 5),
            "等频箱": pd.qcut(v, 5, duplicates="drop"),
        }
    ),
    "银行次数盖帽与分箱",
)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, x, title in zip(
    axes, [v, capped], ["教材银行柜台次数原值", "1%至99%分位盖帽后"]
):
    ax.hist(x, bins=25, color="#82B5CC", edgecolor="white")
    ax.set(title=title, xlabel="柜台服务次数", ylabel="记录数")
c.figure(fig, "盖帽前后分布")
c.notes += [
    "固定示例直接从Notebook原单元格提取，没有自行编写客户、成绩或消费数据。其教学示例身份与配套CSV分开记录。",
    "cell10、95、97的随机数生成不执行；对应行列操作、盖帽、分箱迁移到本章big.csv。盖帽只输出派生列，不改原值，不把盖帽结果当观测。write.csv为原CSV导出示例，输出到作业目录避免覆盖教材。",
]
c.done()
