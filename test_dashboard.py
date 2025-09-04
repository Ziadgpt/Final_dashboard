import pytest
import pandas as pd
from my_db_utils import init_db, save_trades_to_db, load_trades_from_db

def test_db():
    init_db()
    df = pd.DataFrame({
        'symbol': ['PYRUSDT'],
        'trade_num': [1],
        'type': ['Entry long'],
        'datetime': ['2020-01-10 11:08:00'],
        'signal': ['Open'],
        'price_usdt': [10000.0],
        'position_size_qty': [1.0],
        'position_size_value': [10000.0],
        'net_pnl_usdt': [0.0]
    })
    save_trades_to_db(df, 1, 'PYRUSDT', 'test_hash')
    loaded = load_trades_from_db(1, 'PYRUSDT', 'test_hash')
    assert len(loaded) == 1
    assert loaded['symbol'].iloc[0] == 'PYRUSDT'

def test_trade_processing():
    df = pd.DataFrame({
        'Trade #': [1, 1],
        'Type': ['Entry long', 'Exit long'],
        'Date/Time': ['2020-01-10 11:08:00', '2020-01-10 12:08:00'],
        'Signal': ['Open', 'Close'],
        'Price USDT': [10000.0, 10100.0],
        'Position size (qty)': [1.0, 1.0],
        'Position size (value)': [10000.0, 10000.0],
        'Net P&L USDT': [0.0, 100.0]
    })
    trades_list = []
    skipped_trades = []
    for trade_num, group in df.groupby('Trade #'):
        try:
            entry_rows = group[group['Type'].str.lower().str.contains('entry.*long', na=False)]
            exit_rows = group[group['Type'].str.lower().str.contains('exit.*long', na=False)]
            if entry_rows.empty or exit_rows.empty:
                skipped_trades.append((trade_num, f"Missing entry or exit"))
                continue
            trades_list.append({'trade_num': trade_num})
        except Exception as e:
            skipped_trades.append((trade_num, f"Error: {str(e)}"))
    assert len(trades_list) == 1
    assert len(skipped_trades) == 0