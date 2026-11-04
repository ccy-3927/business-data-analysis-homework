"""创建数据模型、快速零售看板与案例1分省销售分析。"""

from chapter2 import *


def main():
    c = newcase(
        __file__,
        "Power BI销售建模",
        "复现教材的关系连接与销售指标；用显式聚合替代 DAX 隐式汇总，用独立图表表达看板。",
    )
    p = source(c, "创建数据模型示例.xlsx")
    fact = excel(p, "销售业绩表")
    d = fact.copy()
    for sheet, key in [
        ("客户信息表", "客户ID"),
        ("城市信息表", "城市编号"),
        ("产品信息表", "产品编号"),
    ]:
        dim = excel(p, sheet)
        assert dim[key].is_unique
        d = d.merge(dim, on=key, how="left", validate="many_to_one")
    assert len(d) == len(fact) and np.isclose(
        d["金额（$M）"].sum(), fact["金额（$M）"].sum()
    )
    c.table(
        d.groupby(["区域名", "产品类型"])["金额（$M）"].sum().unstack(fill_value=0),
        "多维模型销售汇总",
    )
    c.bar(
        d.groupby("区域名")["金额（$M）"].sum().sort_values(ascending=False),
        "区域销售业绩",
        "金额（$M，教材单位）",
    )
    p = source(c, "零售订单表.xlsx")
    retail = excel(p)
    retail["订单时间"] = pd.to_datetime(retail.订单时间)
    profile(c, retail, "零售订单")
    summary = pd.Series(
        {
            "实付金额": retail.实际支付.sum(),
            "去重订单数": retail.订单号.nunique(),
            "商品件数": retail.数量.sum(),
            "客单价": retail.实际支付.sum() / retail.订单号.nunique(),
        }
    )
    c.table(summary, "零售关键指标")
    day = retail.groupby(retail.订单时间.dt.normalize()).实际支付.sum()
    c.line(day, "零售每日实付金额", "金额（原数据单位）", interactive=True)
    category = retail.groupby("商品类别").实际支付.sum().sort_values(ascending=False)
    c.table(category, "品类实付金额")
    c.bar(category, "零售品类贡献", "金额（原数据单位）", interactive=True)
    c.table(
        retail.groupby(["商品类别", "商品名称"]).实际支付.sum(), "零售类别下钻到商品"
    )
    city = retail.groupby("城市").实际支付.sum().sort_values(ascending=False)
    c.table(city, "城市实付金额")
    c.bar(city.head(15), "零售城市销售前15", "金额（原数据单位）")
    base = BASE / "5、PowerBI基本操作/案例1"
    tables = {
        n: csv(c.source(base / (n + ".csv")))
        for n in ["订单表", "订单表明细表", "商品表", "商品类型表", "雇员表"]
    }
    detail = tables["订单表明细表"].rename(
        columns={"订单ID": "OrderID", "产口ID": "ProductID"}
    )
    result = (
        detail.merge(tables["订单表"], on="OrderID", how="left", validate="many_to_one")
        .merge(tables["商品表"], on="ProductID", how="left", validate="many_to_one")
        .merge(
            tables["商品类型表"], on="CategoryID", how="left", validate="many_to_one"
        )
        .merge(tables["雇员表"], on="EmployeeID", how="left", validate="many_to_one")
    )
    result["利润"] = result.销售金额 - result.销售成本
    result["OrderDate"] = pd.to_datetime(result.OrderDate)
    c.table(
        result.groupby("ShipProvince")[["销售金额", "销售成本", "利润"]].sum(),
        "分省销售成本利润",
    )
    province = (
        result.groupby("ShipProvince").销售金额.sum().sort_values(ascending=False)
    )
    c.bar(province, "分省销售金额", "金额（原数据单位）", interactive=True)
    c.table(
        result.groupby("CategoryName")[["销售金额", "利润"]].sum(), "商品类型销售利润"
    )
    c.table(result.groupby("FullName")[["销售金额", "利润"]].sum(), "雇员销售利润")
    product_summary = result.groupby(["CategoryName", "ProductName"]).销售金额.sum()
    c.table(product_summary, "商品类型下钻到商品")
    category_summary = (
        result.groupby("CategoryName").销售金额.sum().sort_values(ascending=False)
    )
    c.bar(category_summary, "分省案例品类销售比较", "金额（原数据单位）")
    c.bar(
        category_summary / category_summary.sum() * 100,
        "分省案例品类销售占比",
        "占比（%）",
    )
    c.table(
        pd.Series(
            {
                "销售金额": result.销售金额.sum(),
                "利润": result.利润.sum(),
                "订单数_订单表计数": len(tables["订单表"]),
                "订单数_明细去重": result.OrderID.nunique(),
            }
        ),
        "分省案例指标卡",
    )
    monthly = result.groupby(
        result.OrderDate.dt.to_period("M").astype(str)
    ).销售金额.sum()
    c.line(monthly, "分省案例月度销售趋势", "金额（原数据单位）")
    checks = pd.Series(
        {
            "原明细行数": len(detail),
            "连接后行数": len(result),
            "未匹配省份行数": result.ShipProvince.isna().sum(),
            "销售额汇总差": result.销售金额.sum() - detail.销售金额.sum(),
            "模型销售总额差": d["金额（$M）"].sum() - fact["金额（$M）"].sum(),
        }
    )
    c.table(checks, "关系与汇总核验")
    assert len(result) == len(detail) and np.isclose(checks["销售额汇总差"], 0)
    c.notes += [
        f"零售订单表含 {len(retail)} 条记录、{retail.订单号.nunique()} 个订单，实付 {retail.实际支付.sum():,.2f}。分类贡献均从明细聚合，没有按行重复累计订单表金额。",
        "分省地图改用可比较的条形图，避免额外获取地图底图；保持省份与销售额的分析口径。商品明细、订单、商品类型、员工均使用明确连接键。品类及员工结果表保留全量，城市图只展示前15且标题明示。",
        f"分省案例关联后 {len(result)} 条明细，未匹配省份 {result.ShipProvince.isna().sum()} 条；销售额连接前后守恒。原列“产口ID”的拼写只在内存规范为 ProductID，不修改源文件。",
    ]
    c.done()


if __name__ == "__main__":
    main()
