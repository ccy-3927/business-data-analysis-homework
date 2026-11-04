"""教材B卡特征函数，修正消费CV误引用余额CV。"""

from model_common import *


def PaymentVar(Balance):
    """
    还款表现类特征
    """
    Bal = Balance.copy()

    ##还款表现特征
    VarSet = {}

    # 账户还款率:账单余额为0或负数时,返回np.nan,即该期还款率计为np.nan
    Bal["Pay_Ratio"] = Bal["Payment"] / Bal["Balance"].map(
        lambda x: np.nan if x <= 0 else x
    )

    ## 还款率_金额_客户维度
    VarSet["Pay_Ratio_AVG_AMT"] = (
        Bal.groupby(["Cust_ID"]).Pay_Ratio.mean().rename("Pay_Ratio_AVG_AMT")
    )  # 期平均
    VarSet["Pay_Ratio_MAX_AMT"] = (
        Bal.groupby(["Cust_ID"]).Pay_Ratio.max().rename("Pay_Ratio_MAX_AMT")
    )  # 单期最大
    VarSet["Pay_Ratio_MIN_AMT"] = (
        Bal.groupby(["Cust_ID"]).Pay_Ratio.min().rename("Pay_Ratio_MIN_AMT")
    )  # 单期最小

    ## 还款率_金额_账户维度
    VarSet["Pay_Ratio_AVG_AMT_ACC"] = (
        Bal.groupby(["Cust_ID", "Account_ID"])
        .Pay_Ratio.mean()
        .groupby("Cust_ID")
        .mean()
        .rename("Pay_Ratio_AVG_AMT_ACC")
    )
    VarSet["Pay_Ratio_MAX_AMT_ACC"] = (
        Bal.groupby(["Cust_ID", "Account_ID"])
        .Pay_Ratio.mean()
        .groupby("Cust_ID")
        .max()
        .rename("Pay_Ratio_MAX_AMT_ACC")
    )
    VarSet["Pay_Ratio_MIN_AMT_ACC"] = (
        Bal.groupby(["Cust_ID", "Account_ID"])
        .Pay_Ratio.mean()
        .groupby("Cust_ID")
        .min()
        .rename("Pay_Ratio_MIN_AMT_ACC")
    )

    ## 还款率_账期_客户维度
    Bal["Pay_Flag"] = Bal.apply(
        lambda x: 0 if x["Payment"] == 0 and x["Balance"] > 0 else 1, 1
    )
    VarSet["Pay_Ratio_Count"] = (
        Bal.groupby(["Cust_ID"]).Pay_Flag.mean().rename("Pay_Ratio_Count")
    )

    ## 还款率_账期_账户维度
    VarSet["Pay_Ratio_Count_AVG_ACC"] = (
        Bal.groupby(["Cust_ID", "Account_ID"])
        .Pay_Flag.mean()
        .groupby("Cust_ID")
        .mean()
        .rename("Pay_Ratio_Count_AVG_ACC")
    )
    VarSet["Pay_Ratio_Count_MAX_ACC"] = (
        Bal.groupby(["Cust_ID", "Account_ID"])
        .Pay_Flag.mean()
        .groupby("Cust_ID")
        .max()
        .rename("Pay_Ratio_Count_MAX_ACC")
    )
    VarSet["Pay_Ratio_Count_MIN_ACC"] = (
        Bal.groupby(["Cust_ID", "Account_ID"])
        .Pay_Flag.mean()
        .groupby("Cust_ID")
        .min()
        .rename("Pay_Ratio_Count_MIN_ACC")
    )

    return VarSet


def DelqVar(Balance):
    """
    逾期类特征
    """
    Bal = Balance.copy()

    ##逾期类特征
    VarSet = {}

    # 总逾期次数_客户维度
    VarSet["Delq1_Count"] = Bal.groupby(["Cust_ID"]).Delq1.sum().rename("Delq1_Count")
    VarSet["Delq2_Count"] = Bal.groupby(["Cust_ID"]).Delq2.sum().rename("Delq2_Count")
    VarSet["Delq3_Count"] = Bal.groupby(["Cust_ID"]).Delq3.sum().rename("Delq3_Count")

    # 总逾期次数占比_客户维度
    VarSet["Delq1_Percent"] = (
        Bal.groupby(["Cust_ID"]).Delq1.mean().rename("Delq1_Percent")
    )
    VarSet["Delq2_Percent"] = (
        Bal.groupby(["Cust_ID"]).Delq2.mean().rename("Delq2_Percent")
    )
    VarSet["Delq3_Percent"] = (
        Bal.groupby(["Cust_ID"]).Delq3.mean().rename("Delq3_Percent")
    )

    # 最高逾期状态_客户维度
    # 20250503，将“1”改为“axis=1”
    VarSet["Delq_Status"] = pd.concat(VarSet, axis=1).apply(
        lambda x: (
            3
            if x["Delq3_Percent"] > 0
            else (2 if x["Delq2_Percent"] > 0 else (1 if x["Delq1_Percent"] > 0 else 0))
        ),
        1,
    )

    # 逾期账户数_账户维度
    VarSet["Delq1_Count_ACC"] = (
        Bal.groupby(["Cust_ID", "Account_ID"])
        .Delq1.max()
        .groupby("Cust_ID")
        .sum()
        .rename("Delq1_Count_ACC")
    )
    VarSet["Delq2_Count_ACC"] = (
        Bal.groupby(["Cust_ID", "Account_ID"])
        .Delq2.max()
        .groupby("Cust_ID")
        .sum()
        .rename("Delq2_Count_ACC")
    )
    VarSet["Delq3_Count_ACC"] = (
        Bal.groupby(["Cust_ID", "Account_ID"])
        .Delq3.max()
        .groupby("Cust_ID")
        .sum()
        .rename("Delq3_Count_ACC")
    )

    return VarSet


def UseVar(BalanceWithLimit):
    """
    额度使用类特征
    """
    Bal = BalanceWithLimit.copy()

    # 当某期账单余额为0或负数时,该期账单余额使用率为np.nan
    Bal["Balance"] = Bal["Balance"].map(lambda x: 0 if x < 0 else x)
    Bal["BillUseRate"] = Bal["Balance"] / Bal["Credit_Limit"]
    Bal["SpendUseRate"] = Bal["Spend"] / Bal["Credit_Limit"]

    ##额度使用类特征
    VarSet = {}

    # 1.使用率
    ##账单额度使用率_客户维度
    VarSet["BillUseRate_AVG"] = (
        Bal.groupby(["Cust_ID"]).BillUseRate.mean().rename("BillUseRate_AVG")
    )
    VarSet["BillUseRate_MAX"] = (
        Bal.groupby(["Cust_ID"]).BillUseRate.max().rename("BillUseRate_MAX")
    )
    VarSet["BillUseRate_MIN"] = (
        Bal.groupby(["Cust_ID"]).BillUseRate.min().rename("BillUseRate_MIN")
    )

    ##消费额度使用率_客户维度
    VarSet["SpendUseRate_AVG"] = (
        Bal.groupby(["Cust_ID"]).SpendUseRate.mean().rename("SpendUseRate_AVG")
    )
    VarSet["SpendUseRate_MAX"] = (
        Bal.groupby(["Cust_ID"]).SpendUseRate.max().rename("SpendUseRate_MAX")
    )
    VarSet["SpendUseRate_MIN"] = (
        Bal.groupby(["Cust_ID"]).SpendUseRate.min().rename("SpendUseRate_MIN")
    )

    ##账户平均账单额度使用率_账户维度
    BillUseRate_Acc = Bal.groupby(["Cust_ID", "Account_ID"]).BillUseRate.mean()
    VarSet["BillUseRate_AVG_ACC"] = (
        BillUseRate_Acc.groupby("Cust_ID").mean().rename("BillUseRate_AVG_ACC")
    )
    VarSet["BillUseRate_MAX_ACC"] = (
        BillUseRate_Acc.groupby("Cust_ID").max().rename("BillUseRate_MAX_ACC")
    )
    VarSet["BillUseRate_MIN_ACC"] = (
        BillUseRate_Acc.groupby("Cust_ID").min().rename("BillUseRate_MIN_ACC")
    )

    ##账户平均消费额度使用率_账户维度
    SpendUseRate_Acc = Bal.groupby(["Cust_ID", "Account_ID"]).SpendUseRate.mean()
    VarSet["SpendUseRate_AVG_ACC"] = (
        SpendUseRate_Acc.groupby("Cust_ID").mean().rename("SpendUseRate_AVG_ACC")
    )
    VarSet["SpendUseRate_MAX_ACC"] = (
        SpendUseRate_Acc.groupby("Cust_ID").max().rename("SpendUseRate_MAX_ACC")
    )
    VarSet["SpendUseRate_MIN_ACC"] = (
        SpendUseRate_Acc.groupby("Cust_ID").min().rename("SpendUseRate_MIN_ACC")
    )

    ##平均额度使用率大于0.75以上的账户数_账户维度
    VarSet["NumAcc_HighBillUseRate"] = (
        BillUseRate_Acc.map(lambda x: 1 if x > 0.75 else 0)
        .groupby("Cust_ID")
        .sum()
        .rename("NumAcc_HighBillUseRate")
    )
    VarSet["NumAcc_HighSpendUseRate"] = (
        SpendUseRate_Acc.map(lambda x: 1 if x > 0.75 else 0)
        .groupby("Cust_ID")
        .sum()
        .rename("NumAcc_HighSpendUseRate")
    )

    ##平均额度使用率小于0.5以下的账户数_账户维度
    VarSet["NumAcc_LowBillUseRate"] = (
        BillUseRate_Acc.map(lambda x: 1 if x < 0.5 else 0)
        .groupby("Cust_ID")
        .sum()
        .rename("NumAcc_LowBillUseRate")
    )
    VarSet["NumAcc_LowSpendUseRate"] = (
        SpendUseRate_Acc.map(lambda x: 1 if x < 0.5 else 0)
        .groupby("Cust_ID")
        .sum()
        .rename("NumAcc_LowSpendUseRate")
    )

    # 2.趋势
    ##账单额度使用率趋势_账户维度:使用最晚期值/往期平滑值近似为趋势,大于1说明使用率趋势增加,小于1说明趋势减少,当分分母为0时(账单余额都为0)时,返回np.nan
    BillUseRate_Trend_ACC = (
        Bal.set_index(["Cust_ID", "Account_ID"]).query("Period==12").BillUseRate
        / Bal.groupby(["Cust_ID", "Account_ID"]).BillUseRate.mean()
    )
    VarSet["BillUseRate_Trend_MAX_ACC"] = (
        BillUseRate_Trend_ACC.groupby("Cust_ID")
        .max()
        .rename("BillUseRate_Trend_MAX_ACC")
    )
    VarSet["BillUseRate_Trend_AVG_ACC"] = (
        BillUseRate_Trend_ACC.groupby("Cust_ID")
        .mean()
        .rename("BillUseRate_Trend_AVG_ACC")
    )
    SpendUseRate_Trend_ACC = (
        Bal.set_index(["Cust_ID", "Account_ID"]).query("Period==12").SpendUseRate
        / Bal.groupby(["Cust_ID", "Account_ID"]).SpendUseRate.mean()
    )
    VarSet["SpendUseRate_Trend_MAX_ACC"] = (
        SpendUseRate_Trend_ACC.groupby("Cust_ID")
        .max()
        .rename("SpendUseRate_Trend_MAX_ACC")
    )
    VarSet["SpendUseRate_Trend_AVG_ACC"] = (
        SpendUseRate_Trend_ACC.groupby("Cust_ID")
        .mean()
        .rename("SpendUseRate_Trend_AVG_ACC")
    )

    # 3.稳定性
    ##额度使用率稳定性_账户维度,当分分母为0时(账单余额都为0)时,稳定性为0
    BillUseRate_CV_ACC = (
        Bal.groupby(["Cust_ID", "Account_ID"]).BillUseRate.std()
        / Bal.groupby(["Cust_ID", "Account_ID"]).BillUseRate.mean()
    ).map(lambda x: 0 if x != x else x)
    VarSet["BillUseRate_CV_AVG_ACC"] = (
        BillUseRate_CV_ACC.groupby("Cust_ID").mean().rename("BillUseRate_CV_AVG_ACC")
    )
    VarSet["BillUseRate_CV_MAX_ACC"] = (
        BillUseRate_CV_ACC.groupby("Cust_ID").max().rename("BillUseRate_CV_MAX_ACC")
    )
    VarSet["BillUseRate_CV_MIN_ACC"] = (
        BillUseRate_CV_ACC.groupby("Cust_ID").min().rename("BillUseRate_CV_MIN_ACC")
    )
    SpendUseRate_CV_ACC = (
        Bal.groupby(["Cust_ID", "Account_ID"]).SpendUseRate.std()
        / Bal.groupby(["Cust_ID", "Account_ID"]).SpendUseRate.mean()
    ).map(lambda x: 0 if x != x else x)
    VarSet["SpendUseRate_CV_AVG_ACC"] = (
        SpendUseRate_CV_ACC.groupby("Cust_ID").mean().rename("SpendUseRate_CV_AVG_ACC")
    )
    VarSet["SpendUseRate_CV_MAX_ACC"] = (
        SpendUseRate_CV_ACC.groupby("Cust_ID").max().rename("SpendUseRate_CV_MAX_ACC")
    )
    VarSet["SpendUseRate_CV_MIN_ACC"] = (
        SpendUseRate_CV_ACC.groupby("Cust_ID").min().rename("SpendUseRate_CV_MIN_ACC")
    )

    ##账户账单余额稳定性_账户维度,当分分母为0时(账单余额都为0)时,稳定性为0
    Bal_CV_ACC = (
        Bal.groupby(["Cust_ID", "Account_ID"]).Balance.std()
        / Bal.groupby(["Cust_ID", "Account_ID"]).Balance.mean()
    ).map(lambda x: 0 if x != x else x)
    VarSet["Balance_CV_Avg"] = (
        Bal_CV_ACC.groupby("Cust_ID").mean().rename("Balance_CV_Avg")
    )
    VarSet["Balance_CV_Max"] = (
        Bal_CV_ACC.groupby("Cust_ID").max().rename("Balance_CV_Max")
    )
    VarSet["Balance_CV_Min"] = (
        Bal_CV_ACC.groupby("Cust_ID").min().rename("Balance_CV_Min")
    )

    ##消费额稳定性_账户维度,当分分母为0时(消费额都为0)时,稳定性为0
    Spend_CV_ACC = (
        Bal.groupby(["Cust_ID", "Account_ID"]).Spend.std()
        / Bal.groupby(["Cust_ID", "Account_ID"]).Spend.mean()
    ).map(lambda x: 0 if x != x else x)
    VarSet["Spend_CV_Avg"] = (
        Spend_CV_ACC.groupby("Cust_ID").mean().rename("Spend_CV_Avg")
    )
    VarSet["Spend_CV_Max"] = (
        Spend_CV_ACC.groupby("Cust_ID").max().rename("Spend_CV_Max")
    )
    VarSet["Spend_CV_Min"] = (
        Spend_CV_ACC.groupby("Cust_ID").min().rename("Spend_CV_Min")
    )

    return VarSet


def AccountInfo(Label):
    """
    账户类基本信息
    """
    lab = Label.copy()

    VarSet = {}

    # 账户信息_客户维度
    VarSet["NumAcc"] = lab.groupby("Cust_ID").Account_ID.count().rename("NumAcc")
    VarSet["CreditLimit_SUM"] = (
        lab.groupby("Cust_ID").Credit_Limit.sum().rename("CreditLimit_SUM")
    )
    VarSet["CreditLimit_MAX"] = (
        lab.groupby("Cust_ID").Credit_Limit.max().rename("CreditLimit_MAX")
    )
    VarSet["CreditLimit_MIN"] = (
        lab.groupby("Cust_ID").Credit_Limit.min().rename("CreditLimit_MIN")
    )

    # 目标信息_客户维度,当客户某账户定义为坏即认定该客户为坏客户
    VarSet["Target"] = lab.groupby("Cust_ID").label.max().rename("Target")

    return VarSet
