import sqlite3
import pandas as pd
from datetime import datetime

def init_db():
    """Initialize SQLite database with trades and metrics tables."""
    try:
        conn = sqlite3.connect("portfolio.db")
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER,
                symbol TEXT,
                trade_num INTEGER,
                type TEXT,
                datetime TEXT,
                signal TEXT,
                price_usdt REAL,
                position_size_qty REAL,
                position_size_value REAL,
                net_pnl_usdt REAL,
                file_hash TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER,
                symbol TEXT,
                net_profit REAL,
                max_drawdown REAL,
                sharpe_ratio REAL,
                sortino_ratio REAL,
                gross_profit REAL,
                gross_loss REAL,
                buy_hold_return REAL,
                max_runup REAL,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                percent_profitable REAL,
                avg_pnl REAL,
                avg_win REAL,
                avg_loss REAL,
                win_loss_ratio REAL,
                avg_bars REAL,
                avg_bars_win REAL,
                avg_bars_loss REAL,
                largest_win REAL,
                largest_loss REAL,
                largest_win_pct REAL,
                largest_loss_pct REAL,
                profit_factor REAL,
                margin_calls INTEGER,
                open_pnl REAL,
                total_open_trades INTEGER,
                timestamp TEXT
            )
        """)
        conn.commit()
    except Exception as e:
        print(f"Error initializing database: {e}")
    finally:
        conn.close()

def save_trades_to_db(df, portfolio_id, symbol, file_hash):
    """Save trades DataFrame to SQLite database."""
    try:
        conn = sqlite3.connect("portfolio.db")
        df['portfolio_id'] = portfolio_id
        df['symbol'] = symbol
        df['file_hash'] = file_hash
        df.to_sql('trades', conn, if_exists='append', index=False)
    except Exception as e:
        print(f"Error saving trades to database: {e}")
    finally:
        conn.close()

def save_metrics_to_db(metrics, portfolio_id, symbol):
    """Save metrics dictionary to SQLite database."""
    try:
        conn = sqlite3.connect("portfolio.db")
        metrics['portfolio_id'] = portfolio_id
        metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        pd.DataFrame([metrics]).to_sql('metrics', conn, if_exists='append', index=False)
    except Exception as e:
        print(f"Error saving metrics to database: {e}")
    finally:
        conn.close()

def load_trades_from_db(portfolio_id, symbol, file_hash):
    """Load trades from SQLite database."""
    try:
        conn = sqlite3.connect("portfolio.db")
        query = "SELECT * FROM trades WHERE portfolio_id = ? AND symbol = ? AND file_hash = ?"
        df = pd.read_sql_query(query, conn, params=(portfolio_id, symbol, file_hash))
        return df
    except Exception as e:
        print(f"Error loading trades from database: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def load_metrics_from_db(portfolio_id, symbol):
    """Load latest metrics from SQLite database."""
    try:
        conn = sqlite3.connect("portfolio.db")
        query = "SELECT * FROM metrics WHERE portfolio_id = ? AND symbol = ? ORDER BY timestamp DESC LIMIT 1"
        df = pd.read_sql_query(query, conn, params=(portfolio_id, symbol))
        return df.to_dict('records')[0] if not df.empty else None
    except Exception as e:
        print(f"Error loading metrics from database: {e}")
        return None
    finally:
        conn.close()