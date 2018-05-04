import pandas as pd

t1 = pd.read_csv("RawData_readytojoin1.csv")
t2 = pd.read_csv("RawData_readytojoin2.csv")
t3 = pd.read_csv("RawData_readytojoin3.csv")
t4 = pd.read_csv("RawData_readytojoin4.csv")
t5 = pd.read_csv("RawData_readytojoin5.csv")
t6 = pd.read_csv("RawData_readytojoin6.csv")

table = t1.merge(t2, 'outer', list(t1)[1]).merge(t3, 'outer', list(t1)[1]).merge(t4, 'outer', list(t1)[1]).merge(t5, 'outer', list(t1)[1]).merge(t6, 'outer', list(t1)[1])

table.to_csv("mergedCurrenciesRaw.csv", sep=',', encoding='utf-8')