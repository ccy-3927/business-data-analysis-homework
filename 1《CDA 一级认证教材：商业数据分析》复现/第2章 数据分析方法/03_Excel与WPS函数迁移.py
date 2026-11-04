"""把教材函数、透视表、条件格式的计算逻辑迁移到 pandas/numpy；不操作 Excel UI。"""

from chapter2 import *
import calendar


def main():
    c = newcase(
        __file__,
        "Excel与WPS函数迁移",
        "按原工作表输入计算 IF/AND/OR、COUNTIF、SUMIF、阶梯计费、日期函数、文本替换、查找、透视表与条件格式。重复的WPS材料合并说明。",
    )
    p = source(c, "1、电子表格常用函数（WPS和EXCEL）.xlsx")
    scores = block(p, "1IF+AND+OR例子", "A1:G13").dropna(subset=["学号"])
    marks = scores[["数学", "计算机", "英语"]].apply(pd.to_numeric)
    scores["三门均通过"] = np.where(marks.ge(60).all(axis=1), "通过", "不通过")
    scores["至少一门通过"] = np.where(marks.ge(60).any(axis=1), "通过", "不通过")
    c.table(scores, "IF_AND_OR成绩判断")
    attendance = block(p, "3COUNTIF-统计", "A1:K33").dropna(subset=["      日期\n学号"])
    attendance = attendance.loc[
        pd.to_numeric(attendance.iloc[:, 0], errors="coerce").notna()
    ]
    attendance["迟到次数"] = (attendance.iloc[:, 1:11] == "b").sum(axis=1)
    attendance["旷课次数"] = (attendance.iloc[:, 1:11] == "c").sum(axis=1)
    c.table(attendance, "COUNTIF出勤统计")
    ledger = block(p, "4借贷-sumif", "A2:I18")
    ledger = ledger.loc[pd.to_numeric(ledger["金额"], errors="coerce").notna()].copy()
    ledger["传票号码"] = ledger["传票号码"].ffill()
    c.table(
        ledger.pivot_table(
            index="传票号码",
            columns="借/贷",
            values="金额",
            aggfunc="sum",
            fill_value=0,
        ),
        "SUMIF借贷汇总",
    )
    usage = block(p, "2IF-水电气费", "A3:D6", ["门牌号", "水量", "电量", "气量"])
    rates = block(p, "2IF-水电气费", "B10:D12", ["第一阶梯", "第二阶梯", "第三阶梯"])

    def tier(x, breaks, prices):
        prices = np.asarray(prices)
        return (
            np.minimum(x, breaks[0]) * prices[0]
            + np.minimum(np.maximum(x - breaks[0], 0), breaks[1] - breaks[0])
            * prices[1]
            + np.maximum(x - breaks[1], 0) * prices[2]
        )

    # 界限是原表 B15:B17、C15:C17 所列的 120/176 与 240/400，不是现实收费标准。
    usage["气费"] = usage.气量 * float(rates.iloc[0, 0])
    usage["电费"] = tier(usage.电量, [240, 400], rates.iloc[1].astype(float))
    usage["水费"] = tier(usage.水量, [120, 176], rates.iloc[2].astype(float))
    usage["总费用"] = usage[["气费", "电费", "水费"]].sum(axis=1)
    c.table(usage, "阶梯水电气费用")
    c.bar(usage.set_index("门牌号")["总费用"], "教材阶梯计费结果", "教材金额单位")
    dates = block(p, "5最后1天等", "A1:C27").dropna(subset=["签订日期"])
    dates["签订日期"] = pd.to_datetime(dates["签订日期"], errors="coerce")
    dates = dates.dropna(subset=["签订日期"])
    dates["该月最后一天"] = dates.签订日期 + pd.offsets.MonthEnd(0)
    dates["该月有多少天"] = dates.签订日期.dt.days_in_month
    c.table(dates, "EOMONTH月末日期")
    d = block(p, "6计算日期", "A1:B21")
    d["日期"] = pd.to_datetime(d.日期)
    d["是否周末"] = d.日期.dt.dayofweek.ge(5)
    c.table(d, "WEEKDAY周末判断")
    holidays = block(p, "6计算日期", "D5:D11", ["假期"]).假期.tolist()
    busday = pd.offsets.CustomBusinessDay(holidays=holidays)
    start = pd.Timestamp(block(p, "6计算日期", "D2:D2", ["日期"]).iloc[0, 0])
    n = int(block(p, "6计算日期", "E2:E2", ["n"]).iloc[0, 0])
    span = block(p, "6计算日期", "H2:I2", ["开始", "结束"]).iloc[0]
    job = block(p, "6计算日期", "G7:I7", ["姓名", "入职日期", "试用月数"]).iloc[0]
    c.table(
        pd.Series(
            {
                "WORKDAY结果": str(start + n * busday),
                "NETWORKDAYS包含首尾": len(
                    pd.date_range(span["开始"], span["结束"], freq=busday)
                ),
                "EDATE转正日期": str(
                    pd.Timestamp(job.入职日期) + pd.DateOffset(months=int(job.试用月数))
                ),
            }
        ),
        "日期函数结果",
    )
    years = block(p, "7MOD判断闰年", "A1:B13").dropna(subset=["年份"])
    years["是否闰年"] = years.年份.astype(int).map(calendar.isleap)
    c.table(years, "MOD闰年判断")
    phones = block(p, "8隐藏几位手机号", "A1:B88").dropna(subset=["手机号"])
    phones["隐藏8-10位"] = (
        phones.手机号.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .map(lambda s: s[:7] + "***" + s[10:])
    )
    c.table(phones[["隐藏8-10位"]], "REPLACE手机号脱敏结果")
    q = source(c, "2、电子表格查找引用函数（WPS和EXCEL）.xlsx")
    sales = block(q, "1.VLOOKUP精确查找", "A1:E3377")
    dim = block(q, "1.VLOOKUP精确查找", "G1:H3377").dropna(subset=["商品编码"])
    c.table(dim.loc[dim.duplicated("商品编码", keep=False)], "商品编码重复映射")
    invalid = dim.商品编码.astype(str).str.startswith("#")
    c.table(dim.loc[invalid], "源表商品编码错误值")
    dim = dim.loc[~invalid].drop_duplicates("商品编码", keep="first")
    sales = sales.drop(columns="商品名称").merge(
        dim, on="商品编码", how="left", validate="many_to_one"
    )
    c.table(sales, "VLOOKUP精确匹配")
    approx = block(q, "1.VLOOKUP近似查找", "A1:D7").dropna(subset=["序号"])
    grades = block(q, "1.VLOOKUP近似查找", "H1:I7").dropna()
    grades = grades.sort_values("评分")
    idx = (
        np.searchsorted(
            grades.评分.to_numpy(), approx.心中评分.to_numpy(), side="right"
        )
        - 1
    )
    assert (idx >= 0).all()
    approx["对应行动"] = grades.行动.to_numpy()[idx]
    c.table(approx, "VLOOKUP近似匹配")
    products = block(q, "3.查找品类名称", "A1:C3315").dropna(subset=["商品名称"])
    cat = block(q, "3.查找品类名称", "E1:F3315").dropna()
    assert cat.品类编号.is_unique
    products["品类名称"] = products.品类编号.map(cat.set_index("品类编号").品类名称)
    c.table(products, "INDEX_MATCH品类查找")
    areas = excel(q, "行政区划分")
    mapping = {col: areas[col].dropna().astype(str).tolist() for col in areas}
    (c.out / "INDIRECT省市映射.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 用源表确实存在的矩形区域展示 OFFSET 的切片语义，不发明员工或区域内容。
    for area in ["D4:G7", "L3:O7", "E16:F18", "J11:K17"]:
        c.table(
            block(
                q,
                "OFFSET-练习自学",
                area,
                [
                    "列" + str(i + 1)
                    for i in range(
                        openpyxl.utils.cell.range_boundaries(area)[2]
                        - openpyxl.utils.cell.range_boundaries(area)[0]
                        + 1
                    )
                ],
            ),
            "OFFSET区域_" + area.replace(":", "_"),
        )
    index_values = block(q, "3.INDEX理解", "B2:H14", list(range(1, 8)))
    c.table(
        pd.Series(
            {
                "INDEX(B2:H14,5,4)": index_values.loc[4, 4],
                "INDEX(B2:H14,6,4)": index_values.loc[5, 4],
            }
        ),
        "INDEX按相对位置取值",
    )
    match_values = block(q, "3.MATCH理解", "F4:L4", list(range(1, 8))).iloc[0]
    target = match_values.dropna().iloc[1]
    c.table(
        pd.Series(
            {"目标": target, "MATCH精确位置": list(match_values).index(target) + 1}
        ),
        "MATCH位置查找",
    )
    p = source(c, "2透视表&条件格式&引用&查找练习.xlsx")
    sales = block(p, "数据透视表", "A1:F43").dropna(subset=["销售地区"])
    assert np.allclose(sales.数量 * sales.单价, sales.销售金额)
    pivot = sales.pivot_table(
        index="销售地区", columns="品名", values="销售金额", aggfunc="sum", fill_value=0
    )
    c.table(pivot, "WPS销售透视表")
    c.heat(pivot, "地区与商品销售额透视")
    grid = block(p, "条件格式", "G3:H12", ["列1", "列2"])
    grid = grid.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    c.table(grid, "条件格式数值")
    c.heat(grid, "条件格式迁移为热力图")
    # 乘法表的输入行列标签均读取原表；乘积是确定性的计算结果，不是模拟数据。
    labels = block(p, "单元格引用", "B1:J1", list(range(9))).iloc[0].astype(int)
    multiplication = pd.DataFrame(
        [[f"{i}×{j}={i*j}" if j <= i else "" for j in labels] for i in labels],
        index=labels,
        columns=labels,
    )
    c.table(multiplication, "混合引用乘法表")
    scores = block(p, "查找", "A1:C41").dropna(subset=["学生姓名"])
    levels = block(p, "查找", "F1:G10").dropna().sort_values("分数")
    pos = np.searchsorted(levels.分数, scores.成绩, side="right") - 1
    assert (pos >= 0).all()
    scores["成绩等级"] = levels.等级.to_numpy()[pos]
    c.table(scores, "成绩区间查找")
    c.notes += [
        "将 IF+AND/OR 迁移为向量条件，COUNTIF 为布尔求和，SUMIF/透视为 groupby/pivot_table，VLOOKUP 精确查找为多对一 merge，近似查找为已排序阈值的 searchsorted，INDEX/MATCH/OFFSET 为位置访问与切片，INDIRECT 级联选择为省市字典。",
        "阶梯计费采用超出部分逐档计价。价格与阈值来自教材，只用于该练习。工作日只排除周末及原表列出的假期，不查询或猜测实际调休。",
        "WPS“1EXCEL常用函数-讲解使用”与这里的 IF、COUNTIF、VLOOKUP 是同类练习，合并实现；函数清单文字和纯鼠标操作不强行作图。条件格式以热力图表达连续数值，不生成 Excel 工作簿。",
        f"品类查找未匹配 {products.品类名称.isna().sum()} 条；透视表金额合计 {pivot.to_numpy().sum():,.0f}，已与明细校验。商品映射中有 {invalid.sum()} 个 Excel 错误值，单列异常并排除错误编码，不将 #VALUE! 当真实商品键；正常重复键按首个匹配项处理。",
    ]
    c.done()


if __name__ == "__main__":
    main()
