"""服装代理商：显式星型模型、经营指标、周环比、店铺/员工目标和会员复购。
教材需求以零售单明细为度量源；订单表只提供维度，不把订单金额复制累加。
"""

from chapter2 import *


def metrics(g):
    amount = g.成交金额.sum()
    retail = g.零售金额.sum()
    qty = g.数量.sum()
    n = g.订单id.nunique()
    members = g.loc[g.会员id.notna()]
    count = members.会员id.nunique()
    member_amount = members.成交金额.sum()
    return pd.Series(
        {
            "成交金额": amount,
            "零售金额": retail,
            "折扣率": amount / retail if retail else np.nan,
            "订单数": n,
            "件数": qty,
            "客单价": amount / n if n else np.nan,
            "件单价": amount / qty if qty else np.nan,
            "连带率": qty / n if n else np.nan,
            "会员人数": count,
            "会员成交金额": member_amount,
            "会员人均消费": member_amount / count if count else np.nan,
        }
    )


def grouped(d, keys):
    records = []
    for key, g in d.groupby(keys, dropna=False, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        records.append(dict(zip(keys, key)) | metrics(g).to_dict())
    return (
        pd.DataFrame(records)
        .set_index(keys)
        .sort_index(key=lambda idx: idx.astype(str))
    )


def main():
    c = newcase(
        __file__,
        "服装零售综合案例",
        "按教材“需求”页从原始多表重算经营指标、目标达成和首次观察消费同期群。独立交互图提供月份切换，保留全量CSV。",
    )
    base = BASE / "5、PowerBI基本操作/案例2"
    p = c.source(base / "服装零售-原始数据.xlsx")
    c.source(base / "【分析思路完整】零售-服装代理商销售分析.xlsx")
    w = pd.read_excel(p, sheet_name=None)
    orders = w["零售单"]
    detail = w["零售单明细"]
    shops = w["店铺表"]
    employees = w["员工表"]
    sku = w["SKU表"]
    styles = w["款号表"]
    attr = w["属性表"]
    for name, key in [
        ("零售单", "订单id"),
        ("零售单明细", "订单明细id"),
        ("店铺表", "店铺id"),
        ("员工表", "员工id"),
        ("SKU表", "SKUid"),
        ("款号表", "款号id"),
        ("属性表", "属性id"),
        ("vip表", "VIPid"),
    ]:
        assert w[name][key].notna().all() and w[name][key].is_unique, name + "主键异常"
    profile(c, detail, "订单明细")
    orders["订单日期"] = pd.to_datetime(orders.订单日期)
    reconciliation = (
        detail.groupby("订单id")[["数量", "零售金额", "成交金额"]]
        .sum()
        .subtract(orders.set_index("订单id")[["数量", "零售金额", "成交金额"]])
    )
    c.table(
        reconciliation.loc[reconciliation.abs().sum(axis=1).gt(0.01)],
        "订单与明细不一致清单",
    )
    assert reconciliation.notna().all().all() and np.allclose(reconciliation, 0)
    enriched = styles.copy()
    for key, name in [
        ("品牌id", "品牌"),
        ("年份id", "款式年份"),
        ("季节id", "季节"),
        ("大类id", "大类"),
    ]:
        enriched = enriched.merge(
            attr.rename(columns={"属性id": key, "属性名称": name}),
            on=key,
            how="left",
            validate="many_to_one",
        )
    sku = (
        sku.merge(enriched, on="款号id", how="left", validate="many_to_one")
        .merge(
            w["颜色表"][["颜色id", "颜色名称"]],
            on="颜色id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            w["尺码表"][["尺码id", "尺码名称"]],
            on="尺码id",
            how="left",
            validate="many_to_one",
        )
    )
    d = detail.merge(
        orders[["订单id", "订单日期", "店铺id", "会员id", "员工id"]],
        on="订单id",
        how="left",
        validate="many_to_one",
    )
    d = (
        d.merge(
            shops[["店铺id", "店铺名称", "店铺面积", "商圈"]],
            on="店铺id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            employees[["员工id", "员工姓名"]],
            on="员工id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            sku, left_on="商品id", right_on="SKUid", how="left", validate="many_to_one"
        )
    )
    assert (
        len(d) == len(detail)
        and d.订单日期.notna().all()
        and np.isclose(d.成交金额.sum(), detail.成交金额.sum())
    )
    d["年月"] = d.订单日期.dt.to_period("M").astype(str)
    d["周起始日"] = d.订单日期.dt.to_period("W-SUN").dt.start_time
    order_qty = d.groupby("订单id").数量.sum()
    d["联单类型"] = np.select(
        [d.订单id.map(order_qty).le(1), d.订单id.map(order_qty).eq(2)],
        ["1件及以下", "2件"],
        default="3件及以上",
    )
    all_metrics = metrics(d)
    c.table(all_metrics, "全期经营指标")
    monthly = grouped(d, ["年月"])
    c.table(monthly, "月度经营指标")
    c.line(
        monthly.成交金额 / 10000,
        "服装零售月度成交趋势",
        "成交金额（万元）",
        interactive=True,
    )
    weekly = grouped(d, ["周起始日"])
    weekly_change = weekly.pct_change(fill_method=None)
    c.table(weekly, "周经营指标")
    c.table(weekly_change, "周环比")
    store = grouped(d, ["店铺id"]).join(
        shops.set_index("店铺id")[["店铺名称", "店铺面积"]]
    )
    store["坪效"] = store.成交金额 / store.店铺面积.replace(0, np.nan)
    c.table(store, "店铺经营指标")
    c.bar(
        store.set_index("店铺名称").成交金额.sort_values(ascending=False) / 10000,
        "店铺成交金额排名",
        "成交金额（万元）",
        interactive=True,
    )
    c.bar(
        store.set_index("店铺名称").坪效.sort_values(ascending=False),
        "店铺全期坪效",
        "全期成交金额 / 店铺面积",
    )
    daily_store = d.pivot_table(
        index="订单日期",
        columns="店铺名称",
        values="成交金额",
        aggfunc="sum",
        fill_value=0,
    )
    daily_eff = daily_store.div(shops.set_index("店铺名称").店铺面积, axis=1)
    c.table(daily_eff, "每日店铺坪效")
    top3 = store.sort_values("成交金额", ascending=False).店铺名称.head(3).tolist()
    last_month = d.订单日期.max().to_period("M")
    focus = daily_eff.loc[daily_eff.index.to_period("M") == last_month, top3]
    c.line(
        focus,
        str(last_month) + "成交前三店每日坪效",
        "当日成交金额 / 店铺面积",
        interactive=True,
    )
    for key in ["品牌", "大类", "颜色名称", "尺码名称", "联单类型", "员工姓名"]:
        tab = grouped(d, [key])
        c.table(tab, key + "经营指标")
    employee_result = grouped(d.loc[d.员工姓名.notna()], ["员工姓名"])
    c.bar(
        employee_result.成交金额.sort_values(ascending=False).head(20) / 10000,
        "员工成交贡献前20",
        "成交金额（万元）",
    )
    c.table(
        pd.Series(
            {"未填写或未匹配员工成交额": d.loc[d.员工姓名.isna(), "成交金额"].sum()}
        ),
        "未归属员工销售额",
    )
    basket = grouped(d, ["联单类型"])
    c.bar(
        basket.成交金额.sort_values(ascending=False) / 10000,
        "不同联单类型成交金额",
        "成交金额（万元）",
    )
    categories = grouped(d, ["大类"])
    c.bar(
        categories.成交金额.sort_values(ascending=False).head(15) / 10000,
        "大类成交贡献前15",
        "成交金额（万元）",
    )
    targets = {}
    for sheet, key in [("店铺月目标表", "店铺id"), ("员工月目标表", "员工id")]:
        target = w[sheet].copy()
        target["年月"] = (
            pd.to_datetime(target.年月.astype(str), format="%Y%m")
            .dt.to_period("M")
            .astype(str)
        )
        assert not target.duplicated([key, "年月"]).any()
        actual = grouped(d, [key, "年月"])
        joined = actual.join(
            target.set_index([key, "年月"]), how="outer", rsuffix="_目标"
        )
        # 观察期内目标存在但无订单的组合，实际值为0；无目标仍留空。
        monthindex = joined.index.get_level_values("年月")
        observed = (monthindex >= d.年月.min()) & (monthindex <= d.年月.max())
        joined.loc[observed, "成交金额"] = joined.loc[observed, "成交金额"].fillna(0)
        for col in ["基础目标", "挑战目标", "攀高目标"]:
            joined[col + "达成率"] = safe_div(joined.成交金额, joined[col])
        c.table(joined, sheet + "_达成率")
        targets[sheet] = joined
    latest = w["店铺月目标表"].年月.max()
    ym = pd.to_datetime(str(latest), format="%Y%m").strftime("%Y-%m")
    target_month = (
        targets["店铺月目标表"]
        .xs(ym, level="年月")
        .join(shops.set_index("店铺id")[["店铺名称"]])
    )
    shown = (
        target_month.loc[target_month.基础目标.notna()]
        .set_index("店铺名称")["基础目标达成率"]
        .sort_values(ascending=False)
        * 100
    )
    c.bar(shown, ym + "店铺基础目标达成", "达成率（%）")
    # 首次购买只在给定观察窗中定义，不使用会员表的最后购买日或总消费额污染历史分群。
    member_orders = orders.loc[
        orders.会员id.notna(), ["订单id", "会员id", "订单日期"]
    ].copy()
    member_orders["消费月"] = member_orders.订单日期.dt.to_period("M")
    first = member_orders.groupby("会员id").消费月.min()
    member_orders["首购月"] = member_orders.会员id.map(first)
    member_orders["第N月"] = member_orders.消费月.astype(
        "int64"
    ) - member_orders.首购月.astype("int64")
    sizes = first.value_counts().sort_index()
    nmonths = int((member_orders.消费月.max() - member_orders.消费月.min()).n) + 1
    rep = (
        member_orders.pivot_table(
            index="首购月", columns="第N月", values="会员id", aggfunc="nunique"
        )
        .reindex(index=sizes.index, columns=range(nmonths))
        .fillna(0)
    )
    for cohort in rep.index:
        rep.loc[cohort, rep.columns > (member_orders.消费月.max() - cohort).n] = np.nan
    rate = rep.div(sizes, axis=0)
    assert rate.max().max() <= 1
    rep.index = rep.index.astype(str)
    rate.index = rate.index.astype(str)
    c.table(rep, "首购同期群人数")
    c.table(rate, "首购同期群复购率")
    show = rate.iloc[:, :12] * 100
    c.heat(show, "首次观察购买同期群_前12个月复购率（%）")
    f = go.Figure(
        go.Heatmap(
            z=rate * 100,
            x=["M" + str(x) for x in rate.columns],
            y=rate.index,
            colorscale="Blues",
            zmin=0,
            zmax=100,
            hovertemplate="首购%{y}<br>%{x}复购%{z:.1f}%<extra></extra>",
            colorbar=dict(title="%"),
        )
    )
    f.update_yaxes(autorange="reversed")
    c.interactive(f, "会员月度复购热力图", static_names=("首次观察购买同期群_前12个月复购率（%）",))
    # 下拉选择真实月份，无插值、无模拟播放帧；用于替代 Power BI 月份切片器。
    by_month = d.pivot_table(
        index="店铺名称", columns="年月", values="成交金额", aggfunc="sum", fill_value=0
    ).sort_index()
    f = go.Figure()
    for i, m in enumerate(by_month):
        f.add_trace(
            go.Bar(
                x=by_month[m] / 10000,
                y=by_month.index,
                orientation="h",
                visible=i == 0,
                name=m,
                hovertemplate="%{y}<br>成交%{x:,.2f}万元<extra></extra>",
            )
        )
    buttons = [
        dict(
            label=m,
            method="update",
            args=[
                {"visible": [j == i for j in range(len(by_month.columns))]},
                {"title": f"店铺月度成交 · {m}"},
            ],
        )
        for i, m in enumerate(by_month.columns)
    ]
    f.update_layout(updatemenus=[dict(buttons=buttons, x=1, y=1.16)], showlegend=False)
    f.update_xaxes(
        title="成交金额（万元）", range=[0, by_month.to_numpy().max() / 10000 * 1.08]
    )
    f.update_yaxes(autorange="reversed")
    c.interactive(f, "店铺成交_月份筛选")
    unmatched = (
        d[["店铺名称", "员工姓名", "品牌", "大类", "颜色名称", "尺码名称"]].isna().sum()
    )
    c.table(unmatched, "维度未匹配或原始缺失数")
    checks = pd.Series(
        {
            "订单数": orders.订单id.nunique(),
            "明细数": len(detail),
            "连接后明细数": len(d),
            "连接前后成交额差": d.成交金额.sum() - detail.成交金额.sum(),
            "会员未在VIP表出现数": d.loc[
                d.会员id.notna() & ~d.会员id.isin(w["vip表"].VIPid), "会员id"
            ].nunique(),
            "目标月份": ym,
        }
    )
    c.table(checks, "结果核验")
    c.notes += [
        f'观察范围 {orders.订单日期.min().date()} 至 {orders.订单日期.max().date()}；订单 {len(orders):,} 笔、明细 {len(detail):,} 行，成交金额 {all_metrics["成交金额"]:,.0f}，客单价 {all_metrics["客单价"]:,.2f}，连带率 {all_metrics["连带率"]:.3f}。订单与明细的数量、零售金额、成交金额逐订单核对一致。',
        "折扣率=成交金额/零售金额，客单价=成交金额/去重订单数，件单价=成交金额/件数，连带率=件数/订单数；会员人均消费只用非空会员的金额和去重会员数。坪效用全期成交金额除单个店铺面积，不能把面积随明细重复求和。",
        "周按周一至周日，环比为本周/上周−1；首末周可能不完整，需注意边界。空员工单列“缺失”组，不假设归属其他员工。联单按订单合计数量分1件及以下/2件/3件及以上，为教材联单维度的明确操作定义。",
        "达成率按 店铺或员工ID＋年月 连接目标，基础/挑战/攀高目标分别计算。没有目标的期间保留空值，不擅自补目标；图展示数据中最后一个目标月。",
        f'员工排名只显示姓名可匹配的记录；未填写或未匹配员工的成交额 {d.loc[d.员工姓名.isna(),"成交金额"].sum():,.0f} 另表披露，不作为一名员工。目标月中最高基础达成率 {shown.max():.2f}%，源目标较低造成的高比率按原值保留。日坪效交互图聚焦 {last_month} 的全期成交前三店，完整各店逐日结果保留CSV。',
        "复购率=该首购同期群在第N月购买的去重会员数/首月新增观察会员数，M0为基期。未完整观察的未来格子留空；样本开始前的购买史未知，因此“首购”仅指数据窗口内首次观察，不能解释为会员终身首次购买。",
        "原文件说明该案例为脱敏教学数据。本地读取原文件，未复制原始教材；字段字典中的发布约束保留供最终发布前核对。",
    ]
    c.done()


if __name__ == "__main__":
    main()
