# Modules
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as plt
from datetime import datetime
import json
from bs4 import BeautifulSoup
import requests

# Parameters
fsym    = "ETH"
tsym    = "USD"
url     = "https://www.cryptocompare.com/api/data/coinsnapshot/?fsym=" + \
          fsym + "&tsym=" + tsym
begin   = "2018-01-01"
end     = "2018-03-13"
dir_out = "../../data/in/"

# Function Defs
def fetchCryptoOHLC_byExchange(fsym, tsym, exchange):
    # a function fetches a crypto OHLC price-series for fsym/tsym and stores
    # it in a pandas DataFrame; uses specific Exchange as provided
    # src: https://www.cryptocompare.com/api/

    cols = ['date', 'timestamp', 'open', 'high', 'low', 'close']
    lst = ['time', 'open', 'high', 'low', 'close']

    timestamp_today = datetime.today().timestamp()
    curr_timestamp = timestamp_today

    for j in range(2):
        df = pd.DataFrame(columns=cols)
        url = "https://min-api.cryptocompare.com/data/histoday?fsym=" + fsym + \
              "&tsym=" + tsym + "&toTs=" + str(int(curr_timestamp)) + \
              "&limit=2000" + "&e=" + exchange
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        dic = json.loads(soup.prettify())

        for i in range(1, 2001):
            tmp = []
            for e in enumerate(lst):
                x = e[0]
                y = dic['Data'][i][e[1]]
                if (x == 0):
                    # timestamp-to-date
                    td = datetime.fromtimestamp(int(y)).strftime('%Y-%m-%d')
                    tmp.append(td)  # (str(timestamp2date(y)))
                tmp.append(y)
            if (np.sum(tmp[-4::]) > 0):
                df.loc[len(df)] = np.array(tmp)
        df.index = pd.to_datetime(df.date)
        df.drop('date', axis=1, inplace=True)
        curr_timestamp = int(df.iloc[0][0])

        if (j == 0):
            df0 = df.copy()
        else:
            data = pd.concat([df, df0], axis=0)

    return data.astype(np.float64)

# Grab Market Names
url = "https://min-api.cryptocompare.com/data/top/exchanges/full?fsym=" + \
       fsym + "&tsym=" + tsym
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
dic = json.loads(soup.prettify())

# Process Names
vol = []
d = dic['Data']['Exchanges']  # a list
for i in range(len(d)):
    vol.append([d[i]['MARKET'], round(float(d[i]['VOLUME24HOUR']), 2)])

# sort a list of sublists according to 2nd item in a sublist
vol = sorted(vol, key=lambda x: -x[1])

# Grab Closing Prices
markets = [e[0] for e in vol][0:10]
if ('cp' in globals()) or ('cp' in locals()): del cp
for market in markets:
    df = fetchCryptoOHLC_byExchange(fsym, tsym, market)
    ts = df[(df.index > begin) & (df.index <= end)]["close"]
    ts.name = market
    if ('cp' in globals()) or ('cp' in locals()):
        cp = pd.concat([cp, ts], axis=1, ignore_index=False)
    else:
        cp = pd.DataFrame(ts)

# Save Data
cp.to_csv(dir_out + "cryptocompare_close.csv")