"""Power Query 拆分、清洗、追加、连接，以及人力与学生函数案例的 Python 实现。"""

from chapter2 import *


def main():
    c = newcase(
        __file__,
        "ETL与函数实操",
        "将 Power Query 表操作映射为字符串处理、类型转换、concat、merge；保留异常记录与缺失，避免自动填造业务数据。",
    )
    p = source(c, "拆分列示例数据.xlsx")
    d = excel(p)
    split = d.assign(订单号=d.订单号.str.split(",")).explode(
        "订单号", ignore_index=True
    )
    split["数量值"] = pd.to_numeric(split.数量.str.extract(r"([0-9.]+)", expand=False))
    split["数量单位"] = split.数量.str.replace(r"[0-9.]+", "", regex=True)
    split["省"] = split.收货地址.str.extract(r"^(.*?省)", expand=False)
    split["客户姓名分词"] = split.客户姓名.str.replace(
        r"([a-z])([A-Z])", r"\1 \2", regex=True
    )
    c.table(split, "按分隔符拆分订单与数量")
    c.notes.append(
        "拆分订单号会复制同行其他字段，这只是 ETL 演示；多个订单共用的数量不可直接按展开后的行累加，否则可能重复。"
    )
    p = source(c, "数据清洗示例数据.xlsx")
    raw = excel(p, 0, header=1)
    profile(c, raw, "清洗前")
    clean = raw.copy()
    clean["源Excel行号"] = clean.index + 3
    for col in clean.select_dtypes("object"):
        clean[col] = clean[col].map(lambda x: x.strip() if isinstance(x, str) else x)
    clean["order_date"] = pd.to_datetime(clean.order_date, errors="coerce")
    for col in ["实付金额", "下单数量"]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean["订单时间"] = pd.to_datetime(
        clean.order_date.dt.strftime("%Y-%m-%d") + " " + clean.order_time.astype(str),
        errors="coerce",
    )
    clean["异常原因"] = ""
    for mask, reason in [
        (clean.订单号.isna(), "缺订单号；"),
        (clean.订单时间.isna(), "时间无法解析；"),
        (clean.实付金额.isna(), "金额缺失；"),
        (clean.下单数量.le(0), "数量非正；"),
        (clean.商品id.isna(), "商品缺失；"),
        (clean.ip地址.astype(str).eq("-911"), "IP字段异常占位；"),
    ]:
        clean.loc[mask, "异常原因"] += reason
    c.table(clean, "标准化数据_不擅自删改业务记录")
    c.table(clean.loc[clean.异常原因.ne("")], "异常记录")
    profile(c, clean, "清洗后")
    for filename in [
        "纵向合并演示-各字段名称一致.xlsx",
        "纵向合并演示-字段名称不一致.xlsx",
    ]:
        p = source(c, filename)
        a = excel(p, "第一季度")
        b = excel(p, "第二季度").rename(columns={"金额": "销售额"})
        merged = pd.concat(
            [a.assign(季度="第一季度"), b.assign(季度="第二季度")], ignore_index=True
        )
        assert len(merged) == len(a) + len(b)
        c.table(merged, Path(filename).stem)
        c.table(merged.groupby("季度").销售额.sum(), Path(filename).stem + "_季度合计")
    p = source(c, "BI横向合并练习文件 .xlsx")
    students = excel(p, "学生表")
    courses = excel(p, "课程表")
    selected = excel(p, "选课表")
    joined = selected.merge(
        students, on="学号", how="left", validate="many_to_one"
    ).merge(courses, on="课程号", how="left", validate="many_to_one")
    assert len(joined) == len(selected)
    c.table(joined.drop(columns=["手机号"], errors="ignore"), "选课学生课程连接结果")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(joined.课程名, joined.成绩, s=55, alpha=0.7)
    ax.set(title="各课程学生成绩（每个点为原始选课记录）", ylabel="成绩")
    ax.grid(axis="y", alpha=0.5)
    c.figure(fig, "课程成绩原始点")
    p = source(c, "考试记录.xlsx")
    exams = excel(p)
    q = source(c, "学生及老师信息.xlsx")
    student = excel(q, "学生信息")
    teachers = excel(q, "教师信息")
    scores = exams.merge(student, on="学号", how="left", validate="one_to_one").merge(
        teachers, on="班级", how="left", validate="many_to_one"
    )
    scores["总分"] = scores[["语文", "数学", "英语"]].sum(axis=1)
    scores["平均分"] = scores[["语文", "数学", "英语"]].mean(axis=1)
    scores["三科均及格"] = scores[["语文", "数学", "英语"]].ge(60).all(axis=1)
    scores["总分名次"] = scores.总分.rank(method="min", ascending=False).astype(int)
    c.table(scores, "学生教师成绩连接与函数")
    c.table(scores.groupby("班级")[["语文", "数学", "英语"]].mean(), "班级科目均分")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(
        [scores[k] for k in ["语文", "数学", "英语"]],
        labels=["语文", "数学", "英语"],
        patch_artist=True,
        boxprops={"facecolor": "#82B5CC"},
    )
    ax.set(title=f"三科考试成绩分布 · 每科 n={len(scores)}", ylabel="分数")
    ax.grid(axis="y", alpha=0.5)
    c.figure(fig, "学生成绩分布")
    p = source(c, "员工信息.xlsx")
    employees = excel(p)
    profile(c, employees, "员工资料")
    for col in ["出生日期", "入职日期", "离职日期"]:
        employees[col] = pd.to_datetime(employees[col], errors="coerce")
    # 入职年龄有明确事件时点，不以今天计算教材历史员工年龄。
    employees["入职时周岁"] = (
        employees.入职日期.dt.year
        - employees.出生日期.dt.year
        - (
            (employees.入职日期.dt.month * 100 + employees.入职日期.dt.day)
            < (employees.出生日期.dt.month * 100 + employees.出生日期.dt.day)
        ).astype(int)
    )
    employees["已离职任职天数"] = (employees.离职日期 - employees.入职日期).dt.days
    c.table(employees, "员工日期函数结果")
    counts = employees.groupby(["部门", "员工状态"]).size().unstack(fill_value=0)
    c.table(counts, "部门在离职人数")
    c.bar(
        employees.groupby("部门").size().sort_values(ascending=False),
        "部门人员数量",
        "记录人数",
    )
    c.notes += [
        "学生例：SUM/AVERAGE/IF/RANK 对应总分、均分、及格判断和并列名次；通过学号、班级连接教师与学生。箱线采用Q1/Q3和1.5IQR须，不附未经要求的显著性检验。",
        "人力例：从明确的出生与入职日期计算入职周岁；仅对已有离职日期计算任职时长，在职人员不凭空指定截止日。部门人数为源表快照，不宣称为当前人数。",
        f'清洗表 {len(raw)} 条，发现 {clean.异常原因.ne("").sum()} 条带异常标记；仅清理文本空格和解析类型，金额缺失不填0。匹配键用 validate 检查连接基数。',
    ]
    c.done()


if __name__ == "__main__":
    main()
