"""读取本教材线性LP文件；不执行文件文本，只解析数值系数。"""

from model_common import *
import re


def read_linear(c, filename):
    p = c.source(c.notebook.parent / filename)
    text = p.read_text(encoding="utf-8-sig")
    section = text[text.index("Maximize") :]
    names = sorted(set(re.findall(r"\bx\d+\b", section)), key=lambda s: int(s[1:]))
    lookup = {name: i for i, name in enumerate(names)}

    def coefficients(expr):
        a = np.zeros(len(names))
        for sign, coef, var in re.findall(r"([+-]?)\s*(\d*\.?\d*)\s*(x\d+)\b", expr):
            a[lookup[var]] += (-1 if sign == "-" else 1) * float(coef or 1)
        return a

    obj = coefficients(re.search(r"obj:(.*?)Subject To", section, re.S).group(1))
    rows = []
    bound = []
    for line in section.split("Subject To")[1].split("Bounds")[0].splitlines():
        if ":" in line:
            lhs, rhs = line.split(":", 1)[1].split("<=")
            rows.append(coefficients(lhs))
            bound.append(float(rhs))
    lower = np.zeros(len(names))
    upper = np.full(len(names), np.inf)
    for line in section.split("Bounds")[1].split("End")[0].splitlines():
        m = re.match(r"\s*([\d.]+)\s*<=\s*(x\d+)(?:\s*<=\s*([\d.]+))?", line)
        if m:
            lower[lookup[m[2]]] = float(m[1])
            upper[lookup[m[2]]] = float(m[3]) if m[3] else np.inf
    return obj, np.array(rows), np.array(bound), lower, upper, names
