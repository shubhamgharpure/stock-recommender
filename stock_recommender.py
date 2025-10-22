import os
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import pandas_ta as ta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from openpyxl import Workbook
import matplotlib.pyplot as plt

try:
    from indstocks import Quote
except ImportError:
    Quote = None

try:
    from financetoolkit import Toolkit
except ImportError:
    Toolkit = None

try:
    from yahoo_fin import stock_info as si
except ImportError:
    si = None

try:
    import talib
except ImportError:
    talib = None

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None

nltk.download('vader_lexicon')

if os.path.exists('stock_data.xlsx'):
    os.remove('stock_data.xlsx')
print("Previous output cleared.")

def load_tickers(file_path):
    with open(file_path, 'r') as f:
        tickers = [line.strip() for line in f.readlines()]
    return [t for t in tickers if t]

yf_tickers = [t + ".NS" for t in load_tickers("tickers.txt")]
plain_tickers = load_tickers("tickers.txt")
print(f"Loaded {len(yf_tickers)} tickers for analysis.")

weights = {
    'fundamental': 0.40,
    'technical': 0.15,
    'historical': 0.10,
    'news': 0.10,
    'quarter': 0.10,
    'momentum': 0.05,
    'liquidity': 0.03,
    'risk': 0.04,
    'dividend': 0.03,
}
extra_weight = 0.03

LOW_FLOAT_THRESHOLD = 1e7          # 10 million shares
LOW_INSIDER_THRESHOLD = 0.10       # 10%, as decimal

recommendations = []
alerts = []
sid = SentimentIntensityAnalyzer()

def get_headlines_from_csv(ticker):
    path = f'news_data/{ticker}.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        return " ".join(df.iloc[:,0].astype(str).tolist())
    return ""

def news_sentiment_fallback(headlines):
    if TextBlob:
        tb_score = TextBlob(headlines).sentiment.polarity
        return tb_score
    elif sid:
        return sid.polarity_scores(headlines)['compound']
    else:
        return 0

for i in tqdm(range(len(yf_tickers))):
    ticker_yf = yf_tickers[i]
    ticker_plain = plain_tickers[i]
    q_revenue_pct = None
    q_income_pct = None
    try:
        stock = yf.Ticker(ticker_yf)
        info = stock.info
        name = info.get('shortName') or ticker_plain
        current_price = info.get('currentPrice', None)
        historical = stock.history(period="2y", interval="1d")
        if historical.empty or historical['Close'].isnull().all():
            raise Exception("No usable historical data found!")
        closes = historical['Close'].dropna()
        old_price = closes.iloc[0] if not closes.empty else None
        recent_price = closes.iloc[-1] if not closes.empty else None
        historical_return = ((recent_price - old_price) / old_price) * 100 if old_price else None

        returns = {}
        for d, label in zip([21, 63, 126], ['1M', '3M', '6M']):
            if len(closes) > d:
                ref_price = closes.iloc[-d]
                returns[label] = ((current_price - ref_price) / ref_price) * 100 if ref_price else None
            else:
                returns[label] = None

        # Bollinger Bands logic: pandas_ta first, fallback to ta-lib (if available)
        try:
            bbands_df = ta.bbands(closes)
            if 'BBU_20_2.0' in bbands_df and 'BBL_20_2.0' in bbands_df:
                bb_upper_series = bbands_df['BBU_20_2.0'].dropna()
                bb_lower_series = bbands_df['BBL_20_2.0'].dropna()
                bb_upper = bb_upper_series.iloc[-1] if not bb_upper_series.empty else None
                bb_lower = bb_lower_series.iloc[-1] if not bb_lower_series.empty else None
            else:
                bb_upper, bb_lower = None, None
        except Exception:
            bb_upper, bb_lower = None, None
        # Fallback to talib
        if (bb_upper is None or bb_lower is None) and talib is not None:
            try:
                upper, middle, lower = talib.BBANDS(closes.values, timeperiod=20)
                bb_upper = upper[-1] if upper is not None and len(upper) > 0 else None
                bb_lower = lower[-1] if lower is not None and len(lower) > 0 else None
            except Exception:
                bb_upper, bb_lower = None, None

        try:
            sma_50 = ta.sma(closes, length=50).iloc[-1]
            sma_200 = ta.sma(closes, length=200).iloc[-1]
        except Exception:
            sma_50, sma_200 = None, None

        try:
            macd_df = ta.macd(closes)
            macd_value = macd_df['MACD_12_26_9'].iloc[-1]
            macd_signal = macd_df['MACDs_12_26_9'].iloc[-1]
        except Exception:
            macd_value, macd_signal = None, None

        try:
            rsi_series = ta.rsi(closes, length=14)
            rsi = float(rsi_series.dropna().iloc[-1]) if not rsi_series.dropna().empty else None
        except Exception:
            rsi = None

        beta = info.get('beta', None)
        stddev_3m = closes.rolling(window=63).std().iloc[-1] if len(closes) >= 63 else None

        pe_ratio = info.get('trailingPE', None)
        debt = info.get('totalDebt', None)
        market_cap = info.get('marketCap', None)
        pb_ratio = info.get('priceToBook', None)
        div_yield = info.get('dividendYield', None)
        roe = info.get('returnOnEquity', None)
        sector = info.get('sector', None)
        held_percent_insiders = info.get('heldPercentInsiders', None)
        float_shares = info.get('floatShares', None)
        avg_volume = info.get('averageVolume', None)
        target_price = info.get('targetMeanPrice', None)
        dividend_rate = info.get('dividendRate', None)
        ex_div_date = info.get('exDividendDate', None)
        earnings_date = info.get('earningsDate', None)
        # New anomaly/peer metrics
        debt_to_equity = info.get('debtToEquity', None)
        eps = info.get('regularMarketEPS', None)
        current_ratio = info.get('currentRatio', None)
        free_cash_flow = info.get('freeCashflow', None)
        week_ago_close = closes.iloc[-6] if len(closes) > 5 else None
        fifty_two_week_high = info.get('fiftyTwoWeekHigh', None)
        fifty_two_week_low = info.get('fiftyTwoWeekLow', None)
        inst_holding = info.get('heldPercentInstitutions', None)
        pe_sector = info.get('sectorPE', None)
        pb_sector = info.get('sectorPB', None)
        roe_sector = info.get('sectorROE', None)

        # Earnings event fallback: yfinance, FinanceToolkit, yahoo_fin
        try:
            qf = stock.quarterly_financials
            if not qf.empty and 'Total Revenue' in qf.index and 'Net Income' in qf.index:
                rev = qf.loc['Total Revenue'].dropna()
                inc = qf.loc['Net Income'].dropna()
                if len(rev) >= 2:
                    q1, q2 = rev.iloc[0], rev.iloc[1]
                    q_revenue_pct = ((q1 - q2) / abs(q2)) * 100 if q2 != 0 else None
                if len(inc) >= 2:
                    iq1, iq2 = inc.iloc[0], inc.iloc[1]
                    q_income_pct = ((iq1 - iq2) / abs(iq2)) * 100 if iq2 != 0 else None
        except Exception as e:
            q_revenue_pct = None
            q_income_pct = None

        if (earnings_date is None or q_revenue_pct is None or q_income_pct is None) and Toolkit is not None:
            try:
                toolkit = Toolkit([ticker_plain])
                df_income = toolkit.get_income_statement(quarterly=True)
                if not df_income.empty:
                    revenues = df_income['Revenue'].dropna()
                    incomes = df_income['Net Income'].dropna()
                    if len(revenues) >= 2 and q_revenue_pct is None:
                        rq1, rq2 = revenues.iloc[0], revenues.iloc[1]
                        q_revenue_pct = ((rq1 - rq2) / abs(rq2)) * 100 if rq2 != 0 else None
                    if len(incomes) >= 2 and q_income_pct is None:
                        iq1, iq2 = incomes.iloc[0], incomes.iloc[1]
                        q_income_pct = ((iq1 - iq2) / abs(iq2)) * 100 if iq2 != 0 else None
            except Exception:
                pass
        if (earnings_date is None or earnings_date == '') and si is not None:
            try:
                earnings = si.get_earnings_history(ticker_yf)
                if earnings and isinstance(earnings, list) and len(earnings) > 0:
                    earnings_date = earnings[0].get('startdatetime', None)
            except Exception:
                pass

        headlines = ""
        news_sentiment_score = 0
        if Quote is not None:
            try:
                quote = Quote(ticker_plain)
                news_list = quote.get_news()
                headlines = " ".join([item['title'] for item in news_list if 'title' in item])
                if headlines.strip():
                    news_sentiment_score = sid.polarity_scores(headlines)['compound']
            except Exception:
                headlines = ""
        if not headlines:
            headlines = get_headlines_from_csv(ticker_plain)
        if not news_sentiment_score and headlines:
            news_sentiment_score = news_sentiment_fallback(headlines)

        anomaly_flag = ""
        # Existing anomalies
        if pe_ratio and pe_ratio > 100: anomaly_flag += "High PE; "
        if roe and roe < 0: anomaly_flag += "Negative ROE; "
        if div_yield is not None and div_yield < 0: anomaly_flag += "Negative Div Yield; "
        # New anomalies
        if debt_to_equity is not None and debt_to_equity > 2: anomaly_flag += "High Debt/Equity; "
        if eps is not None and eps < 0: anomaly_flag += "Negative EPS; "
        if current_ratio is not None and current_ratio < 1: anomaly_flag += "Low Current Ratio; "
        if beta is not None and beta > 2: anomaly_flag += "Highly Volatile (Beta>2); "
        if free_cash_flow is not None and free_cash_flow < 0: anomaly_flag += "Negative Free Cash Flow; "
        if week_ago_close and closes.iloc[-1] < week_ago_close * 0.85: anomaly_flag += "Large Price Drop (>15%/week); "
        if current_price and fifty_two_week_high and current_price >= 0.97 * fifty_two_week_high: anomaly_flag += "Near 52-week High; "
        if current_price and fifty_two_week_low and current_price <= 1.03 * fifty_two_week_low: anomaly_flag += "Near 52-week Low; "
        if inst_holding is not None and inst_holding > 0.5: anomaly_flag += "High Institutional Holding; "
        if pe_sector and pe_ratio and pe_ratio > pe_sector * 1.3: anomaly_flag += "PE much above sector avg; "
        if pb_sector and pb_ratio and pb_ratio > pb_sector * 1.3: anomaly_flag += "PB much above sector avg; "
        if roe_sector and roe and roe < roe_sector * 0.7: anomaly_flag += "ROE much below sector avg; "
        if not headlines or historical.empty: anomaly_flag += "Potential Data Issue; "

        alert_low_float = float_shares is not None and float_shares < LOW_FLOAT_THRESHOLD
        alert_low_insider = held_percent_insiders is not None and held_percent_insiders < LOW_INSIDER_THRESHOLD

        score = 0
        reasons = []
        # (Add scoring as before)
        # -- scoring code here as per earlier version shown above --
        if pe_ratio and 12 <= pe_ratio <= 30: score += weights['fundamental']; reasons.append("Healthy P/E Ratio")
        elif pe_ratio is not None: reasons.append("P/E Ratio out of ideal range")
        if pb_ratio and pb_ratio < 3: score += extra_weight; reasons.append("Healthy P/B Ratio")
        if roe and roe > 0.15: score += extra_weight; reasons.append("High Return on Equity")
        if div_yield and div_yield > 0.01: score += weights['dividend']; reasons.append("Attractive Dividend Yield")
        if rsi is not None and 35 <= rsi <= 65: score += weights['technical']; reasons.append("RSI healthy")
        elif rsi is not None: reasons.append("RSI out of preferred range")
        if sma_50 and sma_200 and sma_50 > sma_200: score += extra_weight; reasons.append("SMA50 above SMA200")
        if macd_value and macd_signal and macd_value > macd_signal: score += extra_weight; reasons.append("Bullish MACD crossover")
        if bb_lower and current_price and current_price < bb_lower*1.05: score += extra_weight; reasons.append("Close near BB lower")
        if historical_return and historical_return > 5: score += weights['historical']; reasons.append("Strong 2-year return")
        elif historical_return is not None: reasons.append("Weak 2-year performance")
        for k, v in returns.items():
            if v is not None and v > 2.5: score += weights['momentum']/3; reasons.append(f"{k} return strong")
            elif v is not None and v < -2.5: reasons.append(f"{k} return negative")
        if beta is not None and beta < 1.3: score += weights['risk']; reasons.append("Acceptable Beta")
        elif beta is not None: reasons.append("High Beta (volatile)")
        if stddev_3m is not None and stddev_3m < 0.07 * current_price: score += weights['risk']/2; reasons.append("Low price stdev")
        if held_percent_insiders and held_percent_insiders > 0.01: score += 0.04; reasons.append("High Insider Holding")
        if float_shares and avg_volume and avg_volume > 100000: score += weights['liquidity']; reasons.append("High liquidity")
        elif avg_volume: reasons.append("Low volume/liquidity")
        if q_revenue_pct is not None and q_revenue_pct > 2.5: score += weights['quarter']/2; reasons.append("Quarterly revenue growth")
        if q_income_pct is not None and q_income_pct > 2.5: score += weights['quarter']/2; reasons.append("Quarterly net income growth")
        if news_sentiment_score > 0.05: score += weights['news']; reasons.append("Positive news sentiment")
        elif news_sentiment_score < -0.05: reasons.append("Negative news sentiment")
        else: reasons.append("Neutral news sentiment")
        if dividend_rate: reasons.append(f"Pays dividend: {dividend_rate}")
        if ex_div_date: reasons.append(f"Ex-Dividend: {ex_div_date}")
        if earnings_date: reasons.append(f"Earnings: {earnings_date}")
        if target_price and current_price and current_price < target_price:
            score += 0.025
            reasons.append(f"Below analyst target ({target_price})")
        if alert_low_float: reasons.append("ALERT: Low float shares (<10M, possible illiquidity)")
        if alert_low_insider: reasons.append("ALERT: Low insider holding (<10%)")
        if alert_low_float or alert_low_insider:
            alerts.append(f"{ticker_plain}: low float={float_shares}, low insider={held_percent_insiders}")

        if score > 0.88 or anomaly_flag:
            alerts.append(f"{ticker_plain}: Score={score:.2f}; Alert: {anomaly_flag}")

        recommendation = "HOLD"
        if score >= 0.75: recommendation = "BUY"
        elif score <= 0.25: recommendation = "SELL"

        recommendations.append({
            "Symbol": ticker_plain,
            "Recommendation": recommendation,
            "Company": name,
            "Current Price": current_price,
            "P/E Ratio": pe_ratio,
            "P/B Ratio": round(pb_ratio, 3) if pb_ratio is not None else None,
            "Div Yield": round(div_yield, 3) if div_yield is not None else None,
            "Dividend Rate": dividend_rate,
            "ROE": round(roe, 3) if roe is not None else None,
            "Debt": debt,
            "Debt to Equity": debt_to_equity,
            "EPS": eps,
            "Current Ratio": current_ratio,
            "Free Cash Flow": free_cash_flow,
            "Market Cap": market_cap,
            "1Y Return %": round(historical_return,2) if historical_return else None,
            "Return_1M": round(returns['1M'],2) if returns['1M'] is not None else None,
            "Return_3M": round(returns['3M'],2) if returns['3M'] is not None else None,
            "Return_6M": round(returns['6M'],2) if returns['6M'] is not None else None,
            "RSI": round(rsi,2) if rsi is not None else None,
            "SMA_50": round(sma_50, 2) if sma_50 is not None else None,
            "SMA_200": round(sma_200, 2) if sma_200 is not None else None,
            "MACD": round(macd_value, 3) if macd_value is not None else None,
            "MACD_Signal": round(macd_signal, 3) if macd_signal is not None else None,
            "BB_Lower": round(bb_lower, 2) if bb_lower is not None else None,
            "BB_Upper": round(bb_upper, 2) if bb_upper is not None else None,
            "Beta": beta,
            "Stddev_3M": round(stddev_3m,2) if stddev_3m is not None else None,
            "Sector": sector,
            "PE Sector": pe_sector,
            "PB Sector": pb_sector,
            "ROE Sector": roe_sector,
            "Insider Holding": held_percent_insiders,
            "Institutional Holding": inst_holding,
            "Float Shares": float_shares,
            "Low Float Alert": alert_low_float,
            "Low Insider Alert": alert_low_insider,
            "Avg Volume": avg_volume,
            "Target Price": target_price,
            "News Sentiment": round(news_sentiment_score,3),
            "Quarterly Revenue %": round(q_revenue_pct,2) if q_revenue_pct is not None else None,
            "Quarterly Net Income %": round(q_income_pct,2) if q_income_pct is not None else None,
            "Event - Earnings": earnings_date,
            "Event - ExDiv": ex_div_date,
            "Weighted Score": round(score,3),
            "Reason": "; ".join(reasons),
            "Anomaly": anomaly_flag,
            "Headlines": headlines,
        })

    except Exception as e:
        print(f"Error fetching data for {ticker_plain}: {e}")
        continue

df = pd.DataFrame(recommendations)
buy_df = df[df['Recommendation']=="BUY"]

# Conviction change analysis
conviction_changes = []
try:
    prev_df = pd.read_excel('stock_data_prev.xlsx') if os.path.exists('stock_data_prev.xlsx') else None
    if prev_df is not None:
        for idx, row in df.iterrows():
            prev_row = prev_df[prev_df['Symbol'] == row['Symbol']]
            if not prev_row.empty:
                prev_rec = prev_row['Recommendation'].values[0]
                prev_score = prev_row['Weighted Score'].values[0]
                if row['Recommendation'] != prev_rec or abs(row['Weighted Score'] - prev_score) > 0.1:
                    conviction_changes.append({
                        "Symbol": row['Symbol'],
                        "Prev Rec": prev_rec,
                        "Cur Rec": row['Recommendation'],
                        "Prev Score": prev_score,
                        "Cur Score": row['Weighted Score'],
                        "Change": row['Weighted Score'] - prev_score,
                        "Direction": "UP" if row['Weighted Score'] > prev_score else "DOWN"
                    })
except Exception as e:
    print("Conviction change analysis skipped:", e)

with pd.ExcelWriter("stock_data.xlsx") as writer:
    df.to_excel(writer, sheet_name="All Recommendations", index=False)
    buy_df[(buy_df["Current Price"]>=100) & (buy_df["Current Price"]<=500)].head(5).to_excel(writer, sheet_name="Buy_Range_100_500", index=False)
    buy_df[(buy_df["Current Price"]>500) & (buy_df["Current Price"]<=1500)].head(5).to_excel(writer, sheet_name="Buy_Range_500_1500", index=False)
    buy_df[(buy_df["Current Price"]>1500) & (buy_df["Current Price"]<=5000)].head(5).to_excel(writer, sheet_name="Buy_Range_1500_5000", index=False)
    df[df["Recommendation"]=="SELL"].to_excel(writer, sheet_name="Sell", index=False)
    df[df["Recommendation"]=="HOLD"].to_excel(writer, sheet_name="Hold", index=False)
    pd.DataFrame({'Alerts': alerts}).to_excel(writer, sheet_name="Alerts", index=False)
    pd.DataFrame(conviction_changes).to_excel(writer, sheet_name="Conviction_Changes", index=False)

# Visualization tab example (top 10 picks trend plot)
try:
    top_picks = buy_df.sort_values("Weighted Score", ascending=False).head(10)
    plt.figure(figsize=(10,6))
    plt.bar(top_picks["Symbol"], top_picks["Weighted Score"])
    plt.xlabel('Symbol')
    plt.ylabel('Weighted Score')
    plt.title('Top 10 Buy Picks Weighted Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("top_picks_chart.png")
    print("Chart saved as top_picks_chart.png.")
except Exception as e:
    print("Chart generation skipped:", e)

print("Analysis complete! Check stock_data.xlsx for recommendations, alerts, conviction changes, anomalies, peer comparison, and visualization.")
