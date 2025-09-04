import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import hashlib
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestRegressor
from my_db_utils import init_db, save_trades_to_db, save_metrics_to_db, load_trades_from_db, load_metrics_from_db
import os
from scipy.optimize import minimize

# Enable debug mode (set to False for production to hide debug logs)
DEBUG_MODE = False

# Page setup
st.set_page_config(page_title="Quant Rebalance Dashboard", layout="wide", initial_sidebar_state="collapsed")
st.title("Smart Portfolio Rebalancer")
st.markdown('<div style="padding-bottom: 20px; color: #FFFFFF;">Analyze your TradingView backtest with advanced metrics, visuals, and AI-driven insights for 1 to 200 symbols.</div>', unsafe_allow_html=True)

# Custom CSS for modern look
st.markdown("""
    <style>
    .main {background-color: #1E1E1E; color: #FFFFFF; padding: 20px;}
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {height: 60px; font-size: 18px; font-weight: bold; background-color: #2A2A2A; border-radius: 10px; padding: 15px 20px; color: #4ECDC4; transition: all 0.3s;}
    .stTabs [data-baseweb="tab"]:hover {background-color: #4ECDC4; color: #1E1E1E;}
    .stTabs [aria-selected="true"] {background-color: #4ECDC4; color: #1E1E1E;}
    .metric-card {background: linear-gradient(135deg, #2A2A2A 0%, #3A3A3A 100%); padding: 20px; border-radius: 12px; margin: 15px 10px; box-shadow: 0 6px 12px rgba(0,0,0,0.3); transition: transform 0.2s;}
    .metric-card:hover {transform: scale(1.05);}
    .metric-title {color: #4ECDC4; font-weight: bold; font-size: 1.1em; margin-bottom: 10px;}
    .metric-value {color: #FFFFFF; font-size: 1.4em;}
    .stPlotlyChart {background-color: #1E1E1E; border-radius: 10px; padding: 15px; margin-bottom: 20px;}
    </style>
""", unsafe_allow_html=True)

# Sidebar
try:
    if os.path.exists("logo.png"):
        st.sidebar.image("logo.png", use_container_width=True, caption="Your Logo")
    else:
        st.sidebar.markdown('<div style="text-align: center; font-weight: bold; color: #4ECDC4;">Logo Placeholder<br>(Add logo.png to directory)</div>', unsafe_allow_html=True)
except Exception as e:
    st.sidebar.markdown(f'<div style="text-align: center; color: #FF6B6B;">Error loading logo: {str(e)}</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div style="font-size: 1.5em; font-weight: bold; color: #4ECDC4; padding-top: 20px;">Smart Portfolio Rebalancer</div>', unsafe_allow_html=True)
st.sidebar.markdown("Free tier: Limited to 1000 trades. Upgrade to [SuperGrok](https://x.ai/grok) for unlimited access.")
st.sidebar.markdown("For API access, visit [x.ai/api](https://x.ai/api).")
with st.sidebar.expander("Help"):
    st.markdown("""
        **How to Use**:
        1. Upload a TradingView backtest Excel file (sheet: 'List of trades').
        2. Select a date range and symbol(s) to filter trades.
        3. Explore tabs: Metrics (key statistics), Visuals (charts), AI Insights (rebalancing, Monte Carlo, anomalies).
        4. Export results as Excel, CSV, or PDF (PDF requires 'reportlab' library, install via 'pip install reportlab').
        **Requirements**: Excel must have columns: Trade #, Type, Date/Time, Signal, Price USDT, Position size (qty), Position size (value), Net P&L USDT.
        **Multi-Symbol**: Include a 'Symbol' column for multiple assets.
    """)
debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=DEBUG_MODE)

# Initialize database
init_db()

# Function to convert Excel serial date or datetime to pandas Timestamp
def excel_serial_to_datetime(value):
    if pd.isna(value):
        return pd.NaT, "NaN value"
    if isinstance(value, datetime):
        return pd.Timestamp(value), None
    try:
        serial = float(value)
        if serial < 0 or serial > 100000:
            return pd.NaT, f"Out-of-range serial: {serial}"
        base_date = datetime(1899, 12, 30)
        delta = timedelta(days=serial)
        return pd.Timestamp(base_date + delta), None
    except (ValueError, TypeError) as e:
        return pd.NaT, f"Conversion error: {str(e)}"

# Compute metrics function (per symbol)
def compute_metrics(portfolio_id: int, symbol: str, trades_df: pd.DataFrame, avg_bars: float):
    try:
        if trades_df.empty:
            if debug_mode:
                st.write(f"Debug: No trades for symbol={symbol}")
            return None, None, None

        for col in ['pnl', 'entry_price', 'exit_price', 'quantity']:
            trades_df[col] = pd.to_numeric(trades_df[col], errors='coerce')
        trades_df.dropna(subset=['pnl', 'entry_price', 'exit_price', 'quantity'], inplace=True)

        if trades_df.empty:
            if debug_mode:
                st.write(f"Debug: All trades for symbol={symbol} dropped due to non-numeric values")
            return None, None, None

        # Sort trades by entry_date to ensure exit_date >= entry_date
        trades_df = trades_df.sort_values('entry_date')
        trades_df['duration'] = (trades_df['exit_date'] - trades_df['entry_date']).dt.total_seconds() / (24 * 60 * 60)
        # Handle same-minute trades by setting minimum duration (1 hour)
        trades_df['duration'] = trades_df['duration'].apply(lambda x: max(x, 1/24) if not pd.isna(x) and x >= 0 else 1/24)

        if debug_mode:
            st.write(f"Debug: {symbol} - Mean duration (days) = {trades_df['duration'].mean():.2f}")
            st.write(f"Debug: {symbol} - Non-zero duration trades = {len(trades_df[trades_df['duration'] > 0])}")
            st.write(f"Debug: {symbol} - Zero or NaN duration trades = {len(trades_df[trades_df['duration'].isna() | (trades_df['duration'] <= 0)])}")
            st.write(f"Debug: {symbol} - Sample entry/exit dates: {trades_df[['entry_date', 'exit_date']].head(2).to_dict()}")

        open_trades = trades_df[trades_df['is_open']]
        open_pnl = (open_trades['quantity'] * (trades_df['entry_price'].iloc[-1] - open_trades['entry_price'])).sum() if not open_trades.empty else 0
        total_open_trades = len(open_trades)

        closed_trades = trades_df[~trades_df['is_open']]
        net_profit = closed_trades['pnl'].sum()
        gross_profit = closed_trades[closed_trades['pnl'] > 0]['pnl'].sum()
        gross_loss = closed_trades[closed_trades['pnl'] < 0]['pnl'].sum()
        equity = closed_trades['pnl'].cumsum() + 100000
        max_runup = (equity - equity.cummin()).max() if len(equity) > 1 else 0
        max_drawdown = ((equity.cummax() - equity) / equity.cummax()).max() * 100 if len(equity) > 1 else 0
        buy_hold_return = (closed_trades['exit_price'].iloc[-1] / closed_trades['entry_price'].iloc[0] - 1) * 100 if len(closed_trades) > 0 and closed_trades['entry_price'].iloc[0] != 0 else 0
        total_trades = len(closed_trades)
        winning_trades = (closed_trades['pnl'] > 0).sum()
        losing_trades = (closed_trades['pnl'] < 0).sum()
        percent_profitable = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        avg_pnl = net_profit / total_trades if total_trades > 0 else 0
        avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0
        win_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 0
        avg_bars_win = closed_trades[closed_trades['pnl'] > 0]['duration'].mean() * 24 if winning_trades > 0 and not pd.isna(closed_trades[closed_trades['pnl'] > 0]['duration'].mean()) else avg_bars
        avg_bars_loss = closed_trades[closed_trades['pnl'] < 0]['duration'].mean() * 24 if losing_trades > 0 and not pd.isna(closed_trades[closed_trades['pnl'] < 0]['duration'].mean()) else avg_bars
        if debug_mode:
            st.write(f"Debug: {symbol} - Avg bars win = {avg_bars_win:.1f}, Avg bars loss = {avg_bars_loss:.1f}")
        largest_win = closed_trades['pnl'].max() if not closed_trades['pnl'].empty else 0.0
        largest_loss = closed_trades['pnl'].min() if not closed_trades['pnl'].empty else 0.0
        largest_win_pct = (largest_win / equity.iloc[0]) * 100 if largest_win and equity.iloc[0] != 0 else 0
        largest_loss_pct = (largest_loss / equity.iloc[0]) * 100 if largest_win and equity.iloc[0] != 0 else 0
        returns = closed_trades['pnl'] / closed_trades['quantity'].mean() if closed_trades['quantity'].mean() != 0 else pd.Series([0])
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        sortino = returns.mean() / returns[returns < 0].std() * np.sqrt(252) if returns[returns < 0].std() != 0 else 0
        profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else 0
        margin_calls = sum(equity < 0.1 * 100000) if len(equity) > 1 else 0

        metrics = {
            'portfolio_id': portfolio_id,
            'symbol': symbol,
            'net_profit': float(net_profit),
            'max_drawdown': float(max_drawdown),
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'gross_profit': float(gross_profit),
            'gross_loss': float(gross_loss),
            'buy_hold_return': float(buy_hold_return),
            'max_runup': float(max_runup),
            'total_trades': int(total_trades),
            'winning_trades': int(winning_trades),
            'losing_trades': int(losing_trades),
            'percent_profitable': float(percent_profitable),
            'avg_pnl': float(avg_pnl),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'win_loss_ratio': float(win_loss_ratio),
            'avg_bars': float(avg_bars),
            'avg_bars_win': float(avg_bars_win),
            'avg_bars_loss': float(avg_bars_loss),
            'largest_win': float(largest_win) if not pd.isna(largest_win) else 0.0,
            'largest_loss': float(largest_loss) if not pd.isna(largest_loss) else 0.0,
            'largest_win_pct': float(largest_win_pct),
            'largest_loss_pct': float(largest_loss_pct),
            'profit_factor': float(profit_factor),
            'margin_calls': int(margin_calls),
            'open_pnl': float(open_pnl),
            'total_open_trades': int(total_open_trades)
        }
        return metrics, closed_trades, equity
    except Exception as e:
        st.error(f"Error computing metrics for {symbol}: {str(e)}")
        return None, None, None

# Monte Carlo simulation
def monte_carlo_simulation(returns, initial_equity=100000, n_simulations=1000, horizon=30):
    try:
        mean_return = returns.mean()
        std_return = returns.std()
        if pd.isna(mean_return) or pd.isna(std_return):
            raise ValueError("Invalid returns data for Monte Carlo simulation")
        simulations = np.random.normal(mean_return, std_return, (horizon, n_simulations))
        paths = initial_equity * np.exp(np.cumsum(simulations, axis=0))
        percentiles = np.percentile(paths, [5, 50, 95], axis=1)
        return paths, percentiles
    except Exception as e:
        st.error(f"Monte Carlo simulation failed: {str(e)}")
        return np.array([]), np.array([])

# Predictive drawdown model
def predict_drawdown(trades_df):
    try:
        if len(trades_df) < 10:
            return None, "Insufficient data for prediction."
        features = trades_df[['duration', 'pnl', 'position_size']].copy()
        features['volatility'] = trades_df['pnl'].rolling(window=5).std()
        features = features.dropna()
        if len(features) < 5:
            return None, "Insufficient data after preprocessing."
        X = features[['duration', 'position_size', 'volatility']]
        y = features['pnl'].shift(-1).dropna()
        X = X.iloc[:-1]
        if len(X) < 5:
            return None, "Too few samples for training."
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        next_features = features[['duration', 'position_size', 'volatility']].iloc[-1].values.reshape(1, -1)
        pred = model.predict(next_features)[0]
        return pred, None
    except Exception as e:
        return None, f"Predictive drawdown failed: {str(e)}"

# Rolling Sharpe ratio
def compute_rolling_sharpe(returns, window=30):
    try:
        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252) if rolling_std.std() != 0 else pd.Series([0] * len(returns))
        return rolling_sharpe
    except Exception as e:
        st.error(f"Rolling Sharpe calculation failed: {str(e)}")
        return pd.Series([0] * len(returns))

# Portfolio optimization (Markowitz)
def portfolio_optimization(returns_df, cov_matrix):
    try:
        n = len(returns_df.columns)
        def objective(weights):
            portfolio_return = np.sum(returns_df.mean() * weights) * 252
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
            return -portfolio_return / portfolio_vol  # Maximize Sharpe
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n))
        initial_guess = [1./n] * n
        result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x
    except Exception as e:
        st.error(f"Portfolio optimization failed: {str(e)}")
        return None

# File upload with loading spinner
with st.spinner("Processing uploaded file..."):
    file = st.file_uploader("Upload your backtest Excel", type=["xlsx"])

# Debug logs in an expander
if debug_mode:
    with st.expander("Debug Logs"):
        debug_container = st.container()

if not file:
    st.warning("Please upload a TradingView backtest Excel file to proceed.")
else:
    # Compute file hash to check for changes
    try:
        file_hash = hashlib.md5(file.getvalue()).hexdigest()
    except Exception as e:
        st.error(f"Error processing uploaded file: {str(e)}")
        st.stop()

    portfolio_id = 1  # Hardcoded for now; add user login later
    default_symbol = 'PYRUSDT'  # Default, overridden if Symbol column exists
    current_date = pd.Timestamp(datetime(2025, 9, 4))

    # Load sheets
    try:
        xl = pd.ExcelFile(file)
        df = pd.read_excel(file, sheet_name="List of trades", dtype={1: str})
        if debug_mode:
            debug_container.markdown(f'<div style="color: #FFFFFF;">Trades loaded: {df.shape[0]} rows, {df.shape[1]} columns</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="color: #FFFFFF;">Trades loaded: {df.shape[0]} rows, {df.shape[1]} columns</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading Excel file: {str(e)}. Ensure it has a 'List of trades' sheet.")
        st.stop()

    # Debug: Show raw column names and sample data
    if debug_mode:
        debug_container.write("Raw column names:", list(df.columns))
        debug_container.write("Sample of first 10 rows:", df.head(10))

    # Dynamically identify columns
    type_col = next((col for col in df.columns if 'type' in col.lower()), df.columns[1] if len(df.columns) > 1 else None)
    trade_num_col = next((col for col in df.columns if 'trade' in col.lower() and 'num' in col.lower()), df.columns[0] if len(df.columns) > 0 else None)
    datetime_col = next((col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()), df.columns[2] if len(df.columns) > 2 else None)
    signal_col = next((col for col in df.columns if 'signal' in col.lower()), df.columns[3] if len(df.columns) > 3 else None)
    price_col = next((col for col in df.columns if 'price' in col.lower() and 'usdt' in col.lower()), df.columns[4] if len(df.columns) > 4 else None)
    pos_size_col = next((col for col in df.columns if 'position' in col.lower() and 'value' in col.lower()), df.columns[6] if len(df.columns) > 6 else None)
    pnl_col = next((col for col in df.columns if 'pnl' in col.lower() and 'usdt' in col.lower()), df.columns[7] if len(df.columns) > 7 else None)
    qty_col = next((col for col in df.columns if 'position' in col.lower() and 'qty' in col.lower()), df.columns[5] if len(df.columns) > 5 else None)

    if not all([type_col, trade_num_col, datetime_col, signal_col, price_col, pos_size_col, pnl_col, qty_col]):
        st.error("Missing required columns. Expected: Trade #, Type, Date/Time, Signal, Price USDT, Position size (qty), Position size (value), Net P&L USDT.")
        if debug_mode:
            debug_container.write("Detected columns:", {"trade_num": trade_num_col, "type": type_col, "datetime": datetime_col,
                                                        "signal": signal_col, "price": price_col, "position_size": pos_size_col,
                                                        "pnl": pnl_col, "quantity": qty_col})
        st.stop()

    if debug_mode:
        debug_container.write(f"Detected columns: trade_num={trade_num_col}, type={type_col}, datetime={datetime_col}, signal={signal_col}, price={price_col}, position_size={pos_size_col}, pnl={pnl_col}, quantity={qty_col}")
        debug_container.write(f"Unique values in '{type_col}' column:", df[type_col].str.strip().dropna().unique().tolist())
        debug_container.write(f"Sample Date/Time values:", df[datetime_col].head(10).tolist())
        debug_container.write(f"Type of Date/Time column:", df[datetime_col].dtype)

    # Check for symbol column
    symbol_col = next((col for col in df.columns if 'symbol' in col.lower()), None)
    if symbol_col:
        if debug_mode:
            debug_container.write(f"Found symbol column: {symbol_col}, unique symbols: {df[symbol_col].unique().tolist()}")
        symbols = df[symbol_col].unique().tolist()
        default_symbol = df[symbol_col].mode()[0] if not df[symbol_col].empty else 'PYRUSDT'
    else:
        st.markdown('<div style="color: #FFFFFF;">No symbol column found. Assuming symbol: PYRUSDT.</div>', unsafe_allow_html=True)
        symbols = ['PYRUSDT']
        default_symbol = 'PYRUSDT'

    # Convert datetime_col to datetime if needed
    if df[datetime_col].dtype != 'datetime64[ns]':
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')

    # Process into trades_df
    trades_list = []
    skipped_trades = []

    # Load cached trades for all symbols
    cached_trades_dict = {symbol: load_trades_from_db(portfolio_id, symbol, file_hash) for symbol in symbols}
    all_cached_empty = all(cached_trades.empty for cached_trades in cached_trades_dict.values())

    if all_cached_empty:
        # Try standard pairing based on type
        type_values = df[type_col].str.strip().str.lower()
        if type_values.str.contains('entry.*long|exit.*long', regex=True, case=False, na=False).any():
            for trade_num, group in df.groupby(trade_num_col):
                try:
                    entry_rows = group[group[type_col].str.strip().str.lower().str.contains('entry.*long', na=False, regex=True)]
                    exit_rows = group[group[type_col].str.strip().str.lower().str.contains('exit.*long', na=False, regex=True)]

                    if entry_rows.empty or exit_rows.empty:
                        skipped_trades.append((trade_num, f"Missing entry ({len(entry_rows)}) or exit ({len(exit_rows)}) row"))
                        continue

                    entry_row = entry_rows.iloc[0]
                    exit_row = exit_rows.iloc[0]

                    entry_date, entry_error = excel_serial_to_datetime(entry_row[datetime_col])
                    if signal_col and 'Open' in str(entry_row[signal_col]) or 'Open' in str(exit_row[signal_col]):
                        exit_date, exit_error = current_date, None
                    else:
                        exit_date, exit_error = excel_serial_to_datetime(exit_row[datetime_col])

                    if pd.isna(entry_date) or pd.isna(exit_date):
                        skipped_trades.append((trade_num, f"Invalid date - entry: {entry_error} (raw: {entry_row[datetime_col]}), exit: {exit_error} (raw: {exit_row[datetime_col]})"))
                        continue

                    trade = {
                        'symbol': entry_row[symbol_col] if symbol_col else default_symbol,
                        'trade_num': int(entry_row[trade_num_col]),
                        'type': entry_row[type_col],
                        'datetime': entry_date,
                        'signal': entry_row[signal_col],
                        'price_usdt': float(entry_row[price_col]),
                        'position_size_qty': float(entry_row[qty_col]),
                        'position_size_value': float(entry_row[pos_size_col]),
                        'net_pnl_usdt': float(exit_row[pnl_col]) if not pd.isna(exit_row[pnl_col]) else 0.0,
                        'entry_price': float(entry_row[price_col]),
                        'exit_price': float(exit_row[price_col]) if not pd.isna(exit_row[price_col]) else entry_row[price_col],
                        'entry_date': entry_date,
                        'exit_date': exit_date,
                        'position_size': float(entry_row[pos_size_col]),
                        'pnl': float(exit_row[pnl_col]) if not pd.isna(exit_row[pnl_col]) else 0.0,
                        'quantity': float(entry_row[qty_col]),
                        'is_open': signal_col and 'Open' in str(exit_row[signal_col]),
                        'bars_in_trade': 0.0
                    }
                    trades_list.append(trade)
                except Exception as e:
                    skipped_trades.append((trade_num, f"Error: {str(e)}"))
                    continue
        else:
            st.warning("No 'entry long' or 'exit long' found in type column. Using fallback: assuming alternating entry/exit rows.")
            for i in range(0, len(df) - 1, 2):
                try:
                    entry_row = df.iloc[i]
                    exit_row = df.iloc[i + 1]
                    trade_num = entry_row[trade_num_col]

                    entry_date, entry_error = excel_serial_to_datetime(entry_row[datetime_col])
                    if signal_col and 'Open' in str(entry_row[signal_col]) or 'Open' in str(exit_row[signal_col]):
                        exit_date, exit_error = current_date, None
                    else:
                        exit_date, exit_error = excel_serial_to_datetime(exit_row[datetime_col])

                    if pd.isna(entry_date) or pd.isna(exit_date):
                        skipped_trades.append((trade_num, f"Invalid date - entry: {entry_error} (raw: {entry_row[datetime_col]}), exit: {exit_error} (raw: {exit_row[datetime_col]})"))
                        continue

                    trade = {
                        'symbol': entry_row[symbol_col] if symbol_col else default_symbol,
                        'trade_num': int(entry_row[trade_num_col]),
                        'type': entry_row[type_col],
                        'datetime': entry_date,
                        'signal': entry_row[signal_col],
                        'price_usdt': float(entry_row[price_col]),
                        'position_size_qty': float(entry_row[qty_col]),
                        'position_size_value': float(entry_row[pos_size_col]),
                        'net_pnl_usdt': float(exit_row[pnl_col]) if not pd.isna(exit_row[pnl_col]) else 0.0,
                        'entry_price': float(entry_row[price_col]),
                        'exit_price': float(exit_row[price_col]) if not pd.isna(exit_row[price_col]) else entry_row[price_col],
                        'entry_date': entry_date,
                        'exit_date': exit_date,
                        'position_size': float(entry_row[pos_size_col]),
                        'pnl': float(exit_row[pnl_col]) if not pd.isna(exit_row[pnl_col]) else 0.0,
                        'quantity': float(entry_row[qty_col]),
                        'is_open': signal_col and 'Open' in str(exit_row[signal_col]),
                        'bars_in_trade': 0.0
                    }
                    trades_list.append(trade)
                except Exception as e:
                    skipped_trades.append((trade_num, f"Error in fallback: {str(e)}"))
                    continue

        trades_df = pd.DataFrame(trades_list)
        if not trades_df.empty:
            trades_df['duration'] = (trades_df['exit_date'] - trades_df['entry_date']).dt.total_seconds() / (24 * 60 * 60)
            trades_df = trades_df.dropna(subset=['duration', 'entry_date', 'exit_date'])
            trades_df = trades_df[trades_df['duration'] >= 0]
            for symbol in trades_df['symbol'].unique():
                save_trades_to_db(trades_df[trades_df['symbol'] == symbol], portfolio_id, symbol, file_hash)
        else:
            st.error("No valid trades processed. Check your Excel data for correct entry/exit pairing.")
            if debug_mode and skipped_trades:
                debug_container.write("Skipped trades (debug info, first 50):", skipped_trades[:50])
            st.stop()
    else:
        trades_df = pd.concat([cached_trades_dict[symbol] for symbol in symbols if not cached_trades_dict[symbol].empty], ignore_index=True)
        trades_df['datetime'] = pd.to_datetime(trades_df['datetime'], errors='coerce')
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'], errors='coerce')
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'], errors='coerce')
        trades_df['duration'] = (trades_df['exit_date'] - trades_df['entry_date']).dt.total_seconds() / (24 * 60 * 60)
        trades_df = trades_df.dropna(subset=['duration', 'entry_date', 'exit_date'])
        trades_df = trades_df[trades_df['duration'] >= 0]

    if trades_df.empty:
        st.error("No valid trades with proper dates found. Check your Excel data.")
        if debug_mode and skipped_trades:
            debug_container.write("Skipped trades (debug info, first 50):", skipped_trades[:50])
        st.stop()

    if debug_mode:
        debug_container.markdown(f'<div style="color: #FFFFFF;">Processed trades: {trades_df.shape[0]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="color: #FFFFFF;">Processed trades: {trades_df.shape[0]}</div>', unsafe_allow_html=True)
    if debug_mode and skipped_trades:
        debug_container.write("Skipped trades (debug info, first 50):", skipped_trades[:50])

    # Load avg_bars
    avg_bars = 0.0
    if "Trades analysis" in xl.sheet_names:
        trades_analysis_df = pd.read_excel(file, sheet_name="Trades analysis")
        if 'Avg # bars in trades' in trades_analysis_df.columns:
            avg_bars = float(trades_analysis_df['Avg # bars in trades'].iloc[0])
            if debug_mode:
                debug_container.markdown(f'<div style="color: #FFFFFF;">Average Bars in Trade from Excel: {avg_bars:.1f}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="color: #FFFFFF;">Average Bars in Trade from Excel: {avg_bars:.1f}</div>', unsafe_allow_html=True)
        else:
            if debug_mode:
                debug_container.markdown('<div style="color: #FFFFFF;">No "Avg # bars in trades" found. Using duration-based estimate.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="color: #FFFFFF;">No "Avg # bars in trades" found. Using duration-based estimate.</div>', unsafe_allow_html=True)
            avg_bars = trades_df['duration'].mean() * 24 if not trades_df['duration'].empty and not pd.isna(trades_df['duration'].mean()) else 10.0
    else:
        if debug_mode:
            debug_container.markdown('<div style="color: #FFFFFF;">No Trades analysis sheet found. Using duration-based estimate.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="color: #FFFFFF;">No Trades analysis sheet found. Using duration-based estimate.</div>', unsafe_allow_html=True)
        avg_bars = trades_df['duration'].mean() * 24 if not trades_df['duration'].empty and not pd.isna(trades_df['duration'].mean()) else 10.0

    # Compute or load metrics for each symbol
    metrics_list = []
    closed_trades_dict = {}
    equity_dict = {}
    for symbol in symbols:
        cached_metrics = load_metrics_from_db(portfolio_id, symbol)
        if cached_metrics:
            metrics = cached_metrics
            closed_trades = trades_df[trades_df['symbol'] == symbol][~trades_df['is_open']]
            equity = closed_trades['pnl'].cumsum() + 100000 if not closed_trades.empty else pd.Series([100000])
        else:
            metrics, closed_trades, equity = compute_metrics(portfolio_id, symbol, trades_df[trades_df['symbol'] == symbol], avg_bars)
            if not metrics:
                st.warning(f"Skipping metrics for {symbol} due to errors.")
                continue
            save_metrics_to_db(metrics, portfolio_id, symbol)
        metrics_list.append(metrics)
        closed_trades_dict[symbol] = closed_trades
        equity_dict[symbol] = equity

    # Symbol filter
    selected_symbols = st.multiselect("Select symbols", symbols, default=symbols)
    if not selected_symbols:
        st.warning("No symbols selected. Please select at least one symbol.")
        st.stop()

    filtered_trades_df = trades_df[trades_df['symbol'].isin(selected_symbols)]
    filtered_metrics = [m for m in metrics_list if m and m['symbol'] in selected_symbols]
    filtered_closed_trades = pd.concat([closed_trades_dict[s] for s in selected_symbols if not closed_trades_dict[s].empty], ignore_index=True)
    filtered_equity = filtered_closed_trades['pnl'].cumsum() + 100000 if not filtered_closed_trades.empty else pd.Series([100000])

    # Date range filter
    date_min = filtered_trades_df['entry_date'].min() if not filtered_trades_df.empty else current_date
    date_max = filtered_trades_df['exit_date'].max() if not filtered_trades_df.empty else current_date
    start_date, end_date = st.date_input("Select date range", [date_min, date_max], min_value=date_min, max_value=date_max)
    filtered_trades_df = filtered_trades_df[(filtered_trades_df['entry_date'] >= pd.Timestamp(start_date)) & (filtered_trades_df['exit_date'] <= pd.Timestamp(end_date))]
    filtered_closed_trades = filtered_closed_trades[(filtered_closed_trades['entry_date'] >= pd.Timestamp(start_date)) & (filtered_closed_trades['exit_date'] <= pd.Timestamp(end_date))]
    filtered_equity = filtered_closed_trades['pnl'].cumsum() + 100000 if not filtered_closed_trades.empty else pd.Series([100000])

    if filtered_trades_df.empty:
        st.warning("No trades match the selected date range or symbols. Adjust your filters.")
        st.stop()

    # Tabs for organization
    tab1, tab2, tab3 = st.tabs(["Metrics", "Visuals", "AI Insights"])

    with tab1:
        st.markdown('<h2 style="color: #4ECDC4; padding-bottom: 20px;">Metrics Summary</h2>', unsafe_allow_html=True)
        try:
            if not filtered_metrics:
                st.markdown('<div style="color: #FFFFFF;">No metrics available for selected symbols.</div>', unsafe_allow_html=True)
            else:
                for metric in filtered_metrics:
                    st.markdown(f'<h3 style="color: #FFFFFF; padding-top: 20px;">{metric["symbol"]}</h3>', unsafe_allow_html=True)
                    cols = st.columns(4)
                    metrics_display = [
                        ("Net Profit (USDT)", f"${metric['net_profit']:,.2f}"),
                        ("Max Drawdown (%)", f"{metric['max_drawdown']:.2f}%"),
                        ("Sharpe Ratio", f"{metric['sharpe_ratio']:.2f}"),
                        ("Sortino Ratio", f"{metric['sortino_ratio']:.2f}"),
                        ("Gross Profit (USDT)", f"${metric['gross_profit']:,.2f}"),
                        ("Gross Loss (USDT)", f"${metric['gross_loss']:,.2f}"),
                        ("Buy and Hold Return (%)", f"{metric['buy_hold_return']:.2f}%"),
                        ("Max Run-up (USDT)", f"${metric['max_runup']:,.2f}"),
                        ("Total Trades", f"{metric['total_trades']}"),
                        ("Winning Trades", f"{metric['winning_trades']}"),
                        ("Losing Trades", f"{metric['losing_trades']}"),
                        ("Percent Profitable (%)", f"{metric['percent_profitable']:.2f}%"),
                        ("Average PnL (USDT)", f"${metric['avg_pnl']:,.2f}"),
                        ("Average Win (USDT)", f"${metric['avg_win']:,.2f}"),
                        ("Average Loss (USDT)", f"${metric['avg_loss']:,.2f}"),
                        ("Win/Loss Ratio", f"{metric['win_loss_ratio']:.2f}"),
                        ("Average Bars in Trade", f"{metric['avg_bars']:.1f}"),
                        ("Average Bars in Winning Trades", f"{metric['avg_bars_win']:.1f}"),
                        ("Average Bars in Losing Trades", f"{metric['avg_bars_loss']:.1f}"),
                        ("Largest Win (USDT)", f"${metric['largest_win']:,.2f}"),
                        ("Largest Loss (USDT)", f"${metric['largest_loss']:,.2f}"),
                        ("Largest Win (%)", f"{metric['largest_win_pct']:.2f}%"),
                        ("Largest Loss (%)", f"{metric['largest_loss_pct']:.2f}%"),
                        ("Profit Factor", f"{metric['profit_factor']:.2f}"),
                        ("Margin Calls", f"{metric['margin_calls']}"),
                        ("Open PnL (USDT)", f"${metric['open_pnl']:,.2f}"),
                        ("Total Open Trades", f"{metric['total_open_trades']}")
                    ]
                    for i, (label, value) in enumerate(metrics_display):
                        with cols[i % 4]:
                            st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-title">{label}</div>
                                    <div class="metric-value">{value}</div>
                                </div>
                            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error rendering Metrics tab: {str(e)}")

    with tab2:
        st.markdown('<h2 style="color: #4ECDC4; padding-bottom: 20px;">Performance Visuals</h2>', unsafe_allow_html=True)
        try:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div style="color: #FFFFFF; padding-bottom: 10px;">Portfolio Weights</div>', unsafe_allow_html=True)
                weights = filtered_trades_df.groupby("symbol")["position_size"].sum()
                weights = weights / weights.sum() if weights.sum() != 0 else weights
                fig = px.pie(values=weights.values, names=weights.index, title="Portfolio Allocation",
                             color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
                fig.update_traces(hovertemplate='%{label}: $%{value:.2f} (%{percent})', textinfo='percent+label')
                fig.update_layout(paper_bgcolor="#1E1E1E", font_color="#FFFFFF", title_font_color="#FFFFFF")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown('<div style="color: #FFFFFF; padding-bottom: 10px;">PnL by Symbol</div>', unsafe_allow_html=True)
                pnl_by_symbol = filtered_closed_trades.groupby("symbol")['pnl'].sum()
                fig = go.Figure(data=[
                    go.Bar(x=pnl_by_symbol.index, y=pnl_by_symbol.values, marker_color='#4ECDC4', hovertemplate='%{x}: $%{y:.2f}<br>')
                ])
                fig.update_layout(title="PnL by Symbol", xaxis_title="Symbol", yaxis_title="PnL (USDT)",
                                  paper_bgcolor="#1E1E1E", plot_bgcolor="#1E1E1E", font_color="#FFFFFF",
                                  title_font_color="#FFFFFF", yaxis_gridcolor="#555555")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div style="color: #FFFFFF; padding-bottom: 10px;">Cumulative PnL Over Time</div>', unsafe_allow_html=True)
            dates = filtered_closed_trades['exit_date'] if not filtered_closed_trades.empty else [current_date] * len(filtered_equity)
            step = max(1, len(dates) // 50)
            downsampled_dates = dates[::step][:len(filtered_equity[::step])]
            downsampled_equity = filtered_equity[::step]
            fig = go.Figure(data=[
                go.Scatter(x=downsampled_dates, y=downsampled_equity, mode='lines', line=dict(color='#4ECDC4'), hovertemplate='Date: %{x}<br>Equity: $%{y:.2f}<br>')
            ])
            fig.update_layout(title="Cumulative PnL", xaxis_title="Date", yaxis_title="Equity (USDT)",
                              paper_bgcolor="#1E1E1E", plot_bgcolor="#1E1E1E", font_color="#FFFFFF",
                              title_font_color="#FFFFFF", xaxis_gridcolor="#555555", yaxis_gridcolor="#555555")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div style="color: #FFFFFF; padding-bottom: 10px;">PnL Distribution</div>', unsafe_allow_html=True)
            fig = px.histogram(filtered_closed_trades, x='pnl', nbins=50, title="PnL Distribution", color_discrete_sequence=['#4ECDC4'])
            fig.add_vline(x=filtered_closed_trades['pnl'].mean() if not filtered_closed_trades.empty else 0, line_dash="dash", line_color="#FF6B6B", annotation_text="Avg PnL")
            fig.update_traces(hovertemplate='PnL: $%{x:.2f}<br>Count: %{y}<br>')
            fig.update_layout(paper_bgcolor="#1E1E1E", plot_bgcolor="#1E1E1E", font_color="#FFFFFF",
                              title_font_color="#FFFFFF", xaxis_gridcolor="#555555", yaxis_gridcolor="#555555")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div style="color: #FFFFFF; padding-bottom: 10px;">Position Size Treemap</div>', unsafe_allow_html=True)
            fig = px.treemap(filtered_trades_df, path=['symbol'], values='position_size', title="Position Size Allocation")
            fig.update_traces(hovertemplate='%{label}: $%{value:.2f}<br>')
            fig.update_layout(paper_bgcolor="#1E1E1E", font_color="#FFFFFF", title_font_color="#FFFFFF")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div style="color: #FFFFFF; padding-bottom: 10px;">Correlation Matrix</div>', unsafe_allow_html=True)
            pivot = filtered_trades_df.pivot_table(index="entry_date", columns="symbol", values="pnl", aggfunc='first').fillna(0)
            corr = pivot.corr()
            fig = px.imshow(corr, title="Correlation Matrix", color_continuous_scale='RdBu', range_color=[-1, 1])
            fig.update_traces(hovertemplate='%{x} vs %{y}: %{z:.2f}<br>')
            fig.update_layout(paper_bgcolor="#1E1E1E", font_color="#FFFFFF", title_font_color="#FFFFFF")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div style="color: #FFFFFF; padding-bottom: 10px;">Drawdown Distribution</div>', unsafe_allow_html=True)
            daily_equity = filtered_closed_trades.groupby(filtered_closed_trades['exit_date'].dt.date)['pnl'].sum().cumsum() + 100000 if not filtered_closed_trades.empty else pd.Series([100000])
            daily_drawdowns = ((daily_equity.cummax() - daily_equity) / daily_equity.cummax()) * 100
            fig = px.box(y=daily_drawdowns, title="Daily Drawdown Distribution", color_discrete_sequence=['#FF6B6B'])
            fig.add_hline(y=metrics_list[0]['max_drawdown'] if metrics_list else 0, line_dash="dash", line_color="#4ECDC4", annotation_text="Max Drawdown")
            fig.update_traces(hovertemplate='Drawdown: %{y:.2f}%<br>')
            fig.update_layout(paper_bgcolor="#1E1E1E", plot_bgcolor="#1E1E1E", font_color="#FFFFFF",
                              title_font_color="#FFFFFF", yaxis_gridcolor="#555555")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div style="color: #FFFFFF; padding-bottom: 10px;">Rolling Sharpe Ratio</div>', unsafe_allow_html=True)
            returns = filtered_closed_trades['pnl'] / filtered_closed_trades['quantity'].mean() if filtered_closed_trades['quantity'].mean() != 0 else pd.Series([0] * len(filtered_closed_trades))
            rolling_sharpe = compute_rolling_sharpe(returns)
            fig = go.Figure(data=[
                go.Scatter(x=filtered_closed_trades['exit_date'], y=rolling_sharpe, mode='lines', line=dict(color='#4ECDC4'), hovertemplate='Date: %{x}<br>Sharpe: %{y:.2f}<br>')
            ])
            fig.update_layout(title="30-Day Rolling Sharpe Ratio", xaxis_title="Date", yaxis_title="Sharpe Ratio",
                              paper_bgcolor="#1E1E1E", plot_bgcolor="#1E1E1E", font_color="#FFFFFF",
                              title_font_color="#FFFFFF", xaxis_gridcolor="#555555", yaxis_gridcolor="#555555")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div style="color: #FFFFFF; padding-bottom: 10px;">Candlestick Chart (Price Trends)</div>', unsafe_allow_html=True)
            for symbol in selected_symbols:
                symbol_trades = filtered_trades_df[filtered_trades_df['symbol'] == symbol]
                if not symbol_trades.empty:
                    fig = go.Figure(data=[
                        go.Candlestick(
                            x=symbol_trades['entry_date'],
                            open=symbol_trades['entry_price'],
                            high=symbol_trades[['entry_price', 'exit_price']].max(axis=1),
                            low=symbol_trades[['entry_price', 'exit_price']].min(axis=1),
                            close=symbol_trades['exit_price'],
                            hoverinfo='x+y+text',
                            text=[f'Open: ${o:.2f}<br>High: ${h:.2f}<br>Low: ${l:.2f}<br>Close: ${c:.2f}'
                                  for o, h, l, c in zip(symbol_trades['entry_price'],
                                                        symbol_trades[['entry_price', 'exit_price']].max(axis=1),
                                                        symbol_trades[['entry_price', 'exit_price']].min(axis=1),
                                                        symbol_trades['exit_price'])]
                        )
                    ])
                    fig.update_layout(title=f"Price Trends for {symbol}", xaxis_title="Date", yaxis_title="Price (USDT)",
                                      paper_bgcolor="#1E1E1E", plot_bgcolor="#1E1E1E", font_color="#FFFFFF",
                                      title_font_color="#FFFFFF", xaxis_gridcolor="#555555", yaxis_gridcolor="#555555")
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering Visuals tab: {str(e)}")

    with tab3:
        st.markdown('<h2 style="color: #4ECDC4; padding-bottom: 20px;">AI Insights</h2>', unsafe_allow_html=True)
        try:
            st.markdown('<div style="color: #FFFFFF; padding-bottom: 10px;">Rebalancing Suggestion</div>', unsafe_allow_html=True)
            returns = filtered_trades_df.groupby("symbol")["pnl"].mean()
            vols = filtered_trades_df.groupby("symbol")["pnl"].std()
            sharpes = returns / vols
            if len(sharpes) > 1:
                best = sharpes.idxmax()
                worst = sharpes.idxmin()
                if sharpes[best] - sharpes[worst] > 0.5:
                    st.markdown(f'<div style="color: #FFFFFF;"><strong>Insight</strong>: Trim {worst} by 10-15% and add to {best}. Last 10 trades suggest {best} reduces risk while maintaining returns.</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="color: #FFFFFF;"><strong>Insight</strong>: Portfolio looks balanced. No major rebalancing needed.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="color: #FFFFFF;"><strong>Insight</strong>: Only one asset ({default_symbol}) detected. Add more assets for rebalancing insights.</div>', unsafe_allow_html=True)

            st.markdown('<div style="color: #FFFFFF; padding-bottom: 10px;">Portfolio Optimization (Markowitz)</div>', unsafe_allow_html=True)
            if len(selected_symbols) > 1:
                returns_df = filtered_trades_df.pivot_table(index="entry_date", columns="symbol", values="pnl", aggfunc='first').fillna(0)
                cov_matrix = returns_df.cov()
                optimal_weights = portfolio_optimization(returns_df, cov_matrix)
                if optimal_weights is not None:
                    weights_df = pd.DataFrame({'Symbol': selected_symbols, 'Optimal Weight': optimal_weights})
                    st.markdown('<div style="color: #FFFFFF;"><strong>Insight</strong>: Optimal portfolio weights to maximize Sharpe ratio:</div>', unsafe_allow_html=True)
                    st.dataframe(weights_df)
                    fig = px.pie(weights_df, values='Optimal Weight', names='Symbol', title="Optimal Portfolio Allocation")
                    fig.update_traces(hovertemplate='%{label}: %{value:.2%}<br>')
                    fig.update_layout(paper_bgcolor="#1E1E1E", font_color="#FFFFFF", title_font_color="#FFFFFF")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown('<div style="color: #FFFFFF;"><strong>Insight</strong>: Portfolio optimization failed due to insufficient data.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="color: #FFFFFF;"><strong>Insight</strong>: Portfolio optimization requires multiple assets.</div>', unsafe_allow_html=True)

            st.markdown('<div style="color: #FFFFFF; padding-bottom: 10px;">Monte Carlo Risk Forecast</div>', unsafe_allow_html=True)
            with st.spinner("Running Monte Carlo simulation..."):
                if st.button("Run Monte Carlo Simulation"):
                    returns = filtered_closed_trades['pnl'] / filtered_closed_trades['quantity'].mean() if filtered_closed_trades['quantity'].mean() != 0 else pd.Series([0])
                    paths, percentiles = monte_carlo_simulation(returns, initial_equity=filtered_equity.iloc[-1] if not filtered_equity.empty else 100000)
                    if paths.size > 0:
                        fig = go.Figure()
                        for path in paths.T[:50]:
                            fig.add_trace(go.Scatter(x=list(range(30)), y=path, mode='lines', line=dict(color='#4ECDC4', width=0.5), opacity=0.1, hovertemplate='Day: %{x}<br>Equity: $%{y:.2f}<br>'))
                        fig.add_trace(go.Scatter(x=list(range(30)), y=percentiles[1], mode='lines', line=dict(color='#FF6B6B'), name='Median', hovertemplate='Day: %{x}<br>Median: $%{y:.2f}<br>'))
                        fig.add_trace(go.Scatter(x=list(range(30)), y=percentiles[2], mode='lines', line=dict(color='#45B7D1', dash='dash'), name='95th Percentile', hovertemplate='Day: %{x}<br>95th: $%{y:.2f}<br>'))
                        fig.add_trace(go.Scatter(x=list(range(30)), y=percentiles[0], mode='lines', line=dict(color='#45B7D1', dash='dash'), name='5th Percentile',
                                                 hovertemplate='Day: %{x}<br>5th: $%{y:.2f}<br>', fill='tonexty', fillcolor='rgba(69, 183, 209, 0.2)'))
                        fig.update_layout(title="Monte Carlo Simulation (30 Days)", xaxis_title="Days", yaxis_title="Equity (USDT)",
                                          paper_bgcolor="#1E1E1E", plot_bgcolor="#1E1E1E", font_color="#FFFFFF",
                                          title_font_color="#FFFFFF", xaxis_gridcolor="#555555", yaxis_gridcolor="#555555")
                        st.plotly_chart(fig, use_container_width=True)
                        var_5 = filtered_equity.iloc[-1] - percentiles[0][-1]
                        st.markdown(f'<div style="color: #FFFFFF;"><strong>Insight</strong>: 5% chance of losing more than ${var_5:,.2f} in 30 days (95% VaR).</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="color: #FFFFFF;"><strong>Insight</strong>: Monte Carlo simulation failed due to insufficient data.</div>', unsafe_allow_html=True)

            st.markdown('<div style="color: #FFFFFF; padding-bottom: 10px;">Anomaly Detection</div>', unsafe_allow_html=True)
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            anomalies = iso_forest.fit_predict(filtered_trades_df[['pnl', 'duration']])
            filtered_trades_df['is_anomaly'] = anomalies == -1
            if filtered_trades_df['is_anomaly'].sum() > 0:
                st.markdown(f'<div style="color: #FFFFFF;"><strong>Insight</strong>: {filtered_trades_df["is_anomaly"].sum()} anomalous trades detected (e.g., extreme PnL or duration).</div>', unsafe_allow_html=True)
                st.dataframe(filtered_trades_df[filtered_trades_df['is_anomaly']][['entry_date', 'pnl', 'duration', 'symbol']])
            else:
                st.markdown('<div style="color: #FFFFFF;"><strong>Insight</strong>: No anomalous trades detected.</div>', unsafe_allow_html=True)

            st.markdown('<div style="color: #FFFFFF; padding-bottom: 10px;">Kelly Criterion Position Sizing</div>', unsafe_allow_html=True)
            if filtered_metrics:
                p = filtered_metrics[0]['percent_profitable'] / 100
                b = filtered_metrics[0]['avg_win'] / abs(filtered_metrics[0]['avg_loss']) if filtered_metrics[0]['avg_loss'] != 0 else 1
                kelly_fraction = (p - (1 - p) / b) if b != 0 else 0
                if kelly_fraction > 0:
                    st.markdown(f'<div style="color: #FFFFFF;"><strong>Insight</strong>: Kelly Criterion suggests allocating {kelly_fraction:.2%} of capital per trade to maximize growth.</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="color: #FFFFFF;"><strong>Insight</strong>: Kelly Criterion not applicable (insufficient win/loss data).</div>', unsafe_allow_html=True)

            st.markdown('<div style="color: #FFFFFF; padding-bottom: 10px;">Predicted Next Drawdown</div>', unsafe_allow_html=True)
            pred, error = predict_drawdown(filtered_trades_df)
            if error:
                st.markdown(f'<div style="color: #FFFFFF;"><strong>Insight</strong>: {error}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="color: #FFFFFF;"><strong>Insight</strong>: Next trade predicted PnL (drawdown risk): ${pred:,.2f} (based on historical patterns).</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error rendering AI Insights tab: {str(e)}")

    # Export options
    st.markdown('<h2 style="color: #4ECDC4; padding-top: 20px;">Export Report</h2>', unsafe_allow_html=True)
    try:
        export_options = ["Excel", "CSV"]
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
            from reportlab.lib import colors
            export_options.append("PDF")
        except ImportError:
            st.warning("PDF export is disabled because 'reportlab' is not installed. Install it with 'pip install reportlab'.")
            letter, SimpleDocTemplate, Table, TableStyle, colors = None, None, None, None, None

        export_format = st.selectbox("Choose format", export_options)
        if st.button("Generate Report"):
            if export_format == "Excel":
                buffer = io.BytesIO()
                filtered_trades_df.to_excel(buffer, index=False)
                st.download_button("Download Excel", buffer.getvalue(), "report.xlsx")
            elif export_format == "CSV":
                buffer = io.StringIO()
                filtered_trades_df.to_csv(buffer, index=False)
                st.download_button("Download CSV", buffer.getvalue(), "report.csv")
            elif export_format == "PDF" and letter is not None:
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=letter)
                elements = []
                table_data = [["Symbol", "Metric", "Value"]] + [
                    [m['symbol'], label, value] for m in filtered_metrics for label, value in [
                        ("Net Profit (USDT)", f"${m['net_profit']:,.2f}"),
                        ("Max Drawdown (%)", f"{m['max_drawdown']:.2f}%"),
                        ("Sharpe Ratio", f"{m['sharpe_ratio']:.2f}"),
                        ("Sortino Ratio", f"{m['sortino_ratio']:.2f}"),
                        ("Gross Profit (USDT)", f"${m['gross_profit']:,.2f}"),
                        ("Gross Loss (USDT)", f"${m['gross_loss']:,.2f}"),
                        ("Buy and Hold Return (%)", f"{m['buy_hold_return']:.2f}%"),
                        ("Max Run-up (USDT)", f"${m['max_runup']:,.2f}"),
                        ("Total Trades", f"{m['total_trades']}"),
                        ("Winning Trades", f"{m['winning_trades']}"),
                        ("Losing Trades", f"{m['losing_trades']}"),
                        ("Percent Profitable (%)", f"{m['percent_profitable']:.2f}%"),
                        ("Average PnL (USDT)", f"${m['avg_pnl']:,.2f}"),
                        ("Average Win (USDT)", f"${m['avg_win']:,.2f}"),
                        ("Average Loss (USDT)", f"${m['avg_loss']:,.2f}"),
                        ("Win/Loss Ratio", f"{m['win_loss_ratio']:.2f}"),
                        ("Average Bars in Trade", f"{m['avg_bars']:.1f}"),
                        ("Average Bars in Winning Trades", f"{m['avg_bars_win']:.1f}"),
                        ("Average Bars in Losing Trades", f"{m['avg_bars_loss']:.1f}"),
                        ("Largest Win (USDT)", f"${m['largest_win']:,.2f}"),
                        ("Largest Loss (USDT)", f"${m['largest_loss']:,.2f}"),
                        ("Largest Win (%)", f"{m['largest_win_pct']:.2f}%"),
                        ("Largest Loss (%)", f"{m['largest_loss_pct']:.2f}%"),
                        ("Profit Factor", f"{m['profit_factor']:.2f}"),
                        ("Margin Calls", f"{m['margin_calls']}"),
                        ("Open PnL (USDT)", f"${m['open_pnl']:,.2f}"),
                        ("Total Open Trades", f"{m['total_open_trades']}")
                    ]
                ]
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(table)
                doc.build(elements)
                st.download_button("Download PDF", buffer.getvalue(), "report.pdf")
            else:
                st.error("PDF export is not available because 'reportlab' is not installed.")
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")