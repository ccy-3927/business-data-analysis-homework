from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parent
paths=['第4章- 数据处理(pandas)\\第4章  数据整合和数据清洗.py', '第5章-数据可视化\\第5章 python数据可视化.py', '第5章-数据可视化\\第5章 描述性统计分析与绘图.py', '第6章-市场调研与数据预处理\\1数据采集与处理 1 数据采集方法\\抽样.py', '第6章-市场调研与数据预处理\\2 数据预处理基础\\数据预处理.py', '第7章-特征降维方法\\在 Python 中实现主成分分析.py', '第7章-特征降维方法\\在 Python 中实现因子分析.py', '第8章-使用统计学方法进行检验和预测\\假设检验与方差分析\\方差分析.py', '第8章-使用统计学方法进行检验和预测\\列联表检验\\列联表分析.py', '第8章-使用统计学方法进行检验和预测\\线性回归\\线性回归.py', '第8章-使用统计学方法进行检验和预测\\逻辑回归\\逻辑回归.py', '第9章-用户分群\\k-means聚类\\Kmeans 的应用 - 用户细分.py', '第9章-用户分群\\层次聚类\\hc_city10.py', '第6章-市场调研与数据预处理\\2 数据预处理基础\\woe.py']
for name in paths:
    subprocess.run([sys.executable,"-X","utf8",str(root/name)],check=True)
