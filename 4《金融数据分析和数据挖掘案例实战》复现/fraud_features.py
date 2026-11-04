"""拟合只发生在训练折的异常特征，以及按日增量的可见网络特征。"""

from model_common import *
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
import networkx as nx


class AnomalyFeatures(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        self.cols = ["edu", "city_level", "gender"]
        self.enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.reg = LinearRegression().fit(
            np.column_stack([x.age, self.enc.fit_transform(x[self.cols])]), x.income
        )
        self.scaler = StandardScaler().fit(x[["age", "income"]])
        resid = x.income - self.reg.predict(
            np.column_stack([x.age, self.enc.transform(x[self.cols])])
        )
        self.rs = StandardScaler().fit(np.asarray(resid).reshape(-1, 1))
        return self

    def transform(self, x):
        out = x.copy()
        out[["age_st", "income_st"]] = self.scaler.transform(x[["age", "income"]])
        out["income_resid"] = x.income - self.reg.predict(
            np.column_stack([x.age, self.enc.transform(x[self.cols])])
        )
        out["income_resid_st"] = self.rs.transform(
            out[["income_resid"]].to_numpy()
        ).ravel()
        for col in ["age_st", "income_st", "income_resid_st"]:
            out[col + "_abs"] = out[col].abs()
        return out.drop(columns=["age", "income"]).fillna(0)


def historical_network(edges):
    g = nx.Graph()
    previous_contacts = set()
    rows = []
    # 已知标签只在其事件日期早于当前申请日时可用，且排除申请人自身。
    events = edges[["userid", "in_date", "flag", "flag_date"]].drop_duplicates()
    for date, day in edges.sort_values("in_date").groupby("in_date"):
        g.add_edges_from(day[["userid", "connid"]].itertuples(index=False, name=None))
        sizes = {}
        risks = {}
        known = events[
            (events.in_date <= date)
            & (events.flag_date < date)
            & events.flag.isin([1, -1, -2])
        ].userid
        known = set(known)
        for component in nx.connected_components(g):
            risk = len(component & known)
            for n in component:
                sizes[n] = len(component)
                risks[n] = risk
        for user in day.userid.unique():
            neighbors = list(g.neighbors(user))
            size = sizes[user]
            rows.append(
                {
                    "userid": user,
                    "date": date,
                    "net_size": size,
                    "degree": g.degree(user),
                    "average_neighbor_degree": (
                        np.mean([g.degree(n) for n in neighbors]) if neighbors else 0
                    ),
                    "percentage": (risks[user] - int(user in known)) / max(1, size - 1),
                    "is_AC": int(user in previous_contacts),
                }
            )
        previous_contacts.update(day.connid)
    return (
        pd.DataFrame(rows)
        .sort_values("date")
        .drop_duplicates("userid", keep="last")
        .set_index("userid")
    )
