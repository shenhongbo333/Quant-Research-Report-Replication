# -*- coding: utf-8 -*-
"""
Hurst 趋势增强 CTA 日频时序策略复现脚本

对应研报：中信期货《期货择时专题报告（六）：如何有效使用赫斯特指数做趋势增强？》

核心逻辑：
1. 获取商品期货主力连续合约日频行情。
2. 计算 Hurst 指数及其相对历史均值的偏离。
3. Hurst 正偏：使用布林带通道突破作为极端趋势信号。
4. Hurst 负偏：使用双均线交叉作为普通趋势信号。
5. 固定交易成本、单日涨跌幅止盈止损、最长持有期。
6. 对每个品种做参数网格搜索，并输出参数敏感性分析图与结果表。
7. 可选：按单品种最优参数做板块/全品种等权、滚动波动率倒数加权组合。

注意：
- 不要把 RQDatac 账号密码硬编码进脚本。建议使用环境变量 RQ_USERNAME / RQ_PASSWORD。
"""

from __future__ import annotations

import argparse
import itertools
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =========================
# 参数区：可直接改这里
# =========================
DEFAULT_SYMBOLS = [
    "CU", "AL", "ZN", "NI",     # 有色金属
    "TA", "RU", "V",             # 能源化工
    "RB", "I", "J", "JM",       # 黑色建材
]

GROUPS = {
    "nonferrous": ["CU", "AL", "ZN", "NI"],
    "energy_chem": ["TA", "RU", "V"],
    "ferrous": ["RB", "I", "J", "JM"],
}

# 研报口径：2周、1月、2月、1季度、4月、5月；半年、7月、8月、3季度；1月、2月、1季度、4月
SHORT_WINDOWS = [10, 20, 40, 60, 80, 100]      # 短均线窗口，也用于布林带窗口
LONG_WINDOWS = [120, 140, 160, 180]            # 长均线窗口，也用于 Hurst 计算窗口
DEV_WINDOWS = [20, 40, 60, 80]                 # Hurst 偏离均值回看窗口

HURST_CONFIGS = {
    # min_chunk 和 n_scales 控制 R/S 回归里的 n 尺度划分；min_segments 控制每个 n 至少切出几段样本。
    "rs_fast": {"min_chunk": 6, "n_scales": 6, "min_segments": 2, "min_valid_scales": 3},
    "rs_default": {"min_chunk": 8, "n_scales": 8, "min_segments": 2, "min_valid_scales": 3},
    "rs_robust": {"min_chunk": 10, "n_scales": 12, "min_segments": 3, "min_valid_scales": 4},
}
DEFAULT_HURST_CONFIGS = ["rs_default"]

TRADING_DAYS = 252
DEFAULT_COST = 1e-4                            # 单边交易成本：万分之一
DEFAULT_BB_K = 1.0                             # 布林带上下轨：1 倍标准差
DEFAULT_STOP_DAILY_RET = 0.05                  # 单日涨跌幅阈值：5%
DEFAULT_MAX_HOLD = 50                          # 最长持有期：50 个交易日


@dataclass
class BacktestResult:
    symbol: str
    short_win: int
    long_win: int
    dev_win: int
    nav: pd.Series
    ret: pd.Series
    position: pd.Series
    metrics: Dict[str, float]
    param_label: str = ""
    hurst_config: str = "rs_default"


# =========================
# RQDatac 数据获取
# =========================
def init_rqdatac(username: Optional[str] = None, password: Optional[str] = None) -> None:
    """初始化 rqdatac。优先读取函数参数，其次读取环境变量。"""
    username = username or os.getenv("RQ_USERNAME")
    password = password or os.getenv("RQ_PASSWORD")
    if not username or not password:
        raise ValueError(
            "缺少 RQDatac 账号密码。请设置环境变量 RQ_USERNAME / RQ_PASSWORD，"
            "或在命令行传入 --username / --password。"
        )
    from rqdatac import init

    print("▶ 初始化 rqdatac...")
    init(username, password)
    print("✅ rqdatac 初始化完成")


def _normalize_rq_price_df(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """统一 rqdatac 返回行情的索引与列。"""
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()

    # futures.get_dominant_price(list) 通常返回 MultiIndex: underlying_symbol/date
    if isinstance(df.index, pd.MultiIndex):
        index_names = [name or "" for name in df.index.names]
        if "underlying_symbol" in index_names:
            try:
                df = df.xs(symbol, level="underlying_symbol")
            except Exception:
                df = df.reset_index()
                df = df[df["underlying_symbol"].astype(str).str.upper() == symbol.upper()]
                if "date" in df.columns:
                    df = df.set_index("date")
        else:
            # 若 MultiIndex 中没有命名，默认最后一层是日期
            df = df.reset_index()
            date_col = "date" if "date" in df.columns else df.columns[-1]
            df = df.set_index(date_col)

    if "date" in df.columns:
        df = df.set_index("date")

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # 部分版本字段大小写/字段缺失不完全一致，这里做容错
    for col in ["open", "high", "low", "close", "settlement", "volume", "open_interest", "total_turnover"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 复现主信号默认用 close；如果你想用 settlement，可在命令行指定 --price-col settlement
    df = df.dropna(subset=["close"])
    return df


def fetch_dominant_daily(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust_type: str = "pre",
    adjust_method: str = "prev_close_ratio",
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """获取单个期货品种主力连续合约日线。优先使用 futures.get_dominant_price。"""
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{symbol}_{start_date}_{end_date}_{adjust_type}_{adjust_method}.csv"
        if cache_path.exists() and not force_download:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return df

    from rqdatac import futures

    fields = [
        "open", "high", "low", "close", "settlement", "prev_settlement",
        "volume", "open_interest", "total_turnover", "limit_up", "limit_down", "dominant_id"
    ]
    try:
        raw = futures.get_dominant_price(
            underlying_symbols=symbol,
            start_date=start_date,
            end_date=end_date,
            frequency="1d",
            fields=fields,
            adjust_type=adjust_type,
            adjust_method=adjust_method,
            rule=0,
            rank=1,
        )
    except Exception as e1:
        print(f"⚠️ {symbol}: futures.get_dominant_price 带 dominant_id 失败，尝试减少字段。原因：{e1}")
        fields2 = [
            "open", "high", "low", "close", "settlement", "prev_settlement",
            "volume", "open_interest", "total_turnover", "limit_up", "limit_down"
        ]
        raw = futures.get_dominant_price(
            underlying_symbols=symbol,
            start_date=start_date,
            end_date=end_date,
            frequency="1d",
            fields=fields2,
            adjust_type=adjust_type,
            adjust_method=adjust_method,
            rule=0,
            rank=1,
        )

    df = _normalize_rq_price_df(raw, symbol)
    if df.empty:
        raise ValueError(f"{symbol} 没有取到行情数据，请检查品种代码、权限或日期范围。")

    if cache_dir is not None:
        df.to_csv(cache_path, encoding="utf-8-sig")
    return df


def load_all_data(
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    data_dir: Path,
    adjust_type: str,
    adjust_method: str,
    force_download: bool,
) -> Dict[str, pd.DataFrame]:
    data = {}
    for symbol in symbols:
        print(f"▶ 获取 {symbol} 主力连续日线...")
        try:
            df = fetch_dominant_daily(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust_type=adjust_type,
                adjust_method=adjust_method,
                cache_dir=data_dir,
                force_download=force_download,
            )
            print(f"✅ {symbol}: {df.index.min().date()} ~ {df.index.max().date()}，{len(df)} 行")
            data[symbol] = df
        except Exception as e:
            print(f"❌ {symbol} 获取失败：{e}")
    return data


# =========================
# Hurst 计算
# =========================
def hurst_rs(
    series: np.ndarray,
    min_chunk: int = 8,
    n_scales: int = 8,
    min_segments: int = 2,
    min_valid_scales: int = 3,
) -> float:
    """
    用重标极差 R/S 方法估计 Hurst 指数。

    公式：log(R/S)_n = log(K) + H * log(n)
    对不同 n 的 log(n), log(R/S) 做 OLS，斜率为 H。

    这里为了稳定性，对价格取 log 后做 R/S。若价格中有非正数，则返回 NaN。
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 40 or np.any(x <= 0):
        return np.nan

    x = np.log(x)
    N = len(x)
    max_chunk = N // 2
    if max_chunk <= min_chunk:
        return np.nan

    scales = np.unique(np.floor(np.geomspace(min_chunk, max_chunk, n_scales)).astype(int))
    rs_values = []
    valid_scales = []

    for n in scales:
        if n < 4:
            continue
        a = N // n
        if a < min_segments:
            continue

        rs_list = []
        for k in range(a):
            seg = x[k * n:(k + 1) * n]
            if len(seg) < n:
                continue
            mean = np.mean(seg)
            dev = seg - mean
            cum_dev = np.cumsum(dev)
            R = np.max(cum_dev) - np.min(cum_dev)
            S = np.std(seg, ddof=1)
            if S > 0 and np.isfinite(R) and np.isfinite(S):
                rs_list.append(R / S)
        if len(rs_list) > 0:
            rs_mean = np.mean(rs_list)
            if rs_mean > 0 and np.isfinite(rs_mean):
                valid_scales.append(n)
                rs_values.append(rs_mean)

    if len(valid_scales) < min_valid_scales:
        return np.nan

    log_n = np.log(np.asarray(valid_scales, dtype=float))
    log_rs = np.log(np.asarray(rs_values, dtype=float))
    slope, _ = np.polyfit(log_n, log_rs, 1)
    return float(slope)


def rolling_hurst(price: pd.Series, window: int, hurst_config: str, config: Dict[str, int]) -> pd.Series:
    """滚动计算 Hurst，window 通常为 120/140/160/180。"""
    arr = price.astype(float).values
    out = np.full(len(arr), np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        out[i] = hurst_rs(arr[i - window + 1:i + 1], **config)
    return pd.Series(out, index=price.index, name=f"hurst_{window}_{hurst_config}")


def precompute_hurst(
    price: pd.Series,
    long_windows: Iterable[int],
    hurst_configs: Iterable[str],
) -> Dict[Tuple[int, str], pd.Series]:
    """对每个 long_window 和 Hurst 估计口径预计算 Hurst，避免网格搜索重复计算。"""
    result = {}
    for w, cfg_name in itertools.product(sorted(set(long_windows)), hurst_configs):
        cfg = HURST_CONFIGS[cfg_name]
        print(f"  ▶ 预计算 Hurst window={w}, config={cfg_name}")
        result[(w, cfg_name)] = rolling_hurst(price, w, cfg_name, cfg)
    return result


# =========================
# 策略回测
# =========================
def calc_metrics(ret: pd.Series, nav: Optional[pd.Series] = None) -> Dict[str, float]:
    ret = ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if nav is None:
        nav = (1.0 + ret).cumprod()
    nav = nav.replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)

    n = len(ret)
    if n == 0:
        return {
            "ann_ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan,
            "max_drawdown": np.nan, "calmar": np.nan, "win_rate": np.nan,
            "profit_loss_ratio": np.nan, "turnover": np.nan,
        }

    ann_ret = nav.iloc[-1] ** (TRADING_DAYS / max(n, 1)) - 1.0
    ann_vol = ret.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = ann_ret / ann_vol if ann_vol and ann_vol > 0 else np.nan

    running_max = nav.cummax()
    dd = nav / running_max - 1.0
    max_dd = -dd.min() if len(dd) > 0 else np.nan
    calmar = ann_ret / max_dd if max_dd and max_dd > 0 else np.nan

    active_ret = ret[ret != 0]
    win_rate = (active_ret > 0).mean() if len(active_ret) > 0 else np.nan
    pos_mean = active_ret[active_ret > 0].mean()
    neg_mean = active_ret[active_ret < 0].mean()
    pl_ratio = pos_mean / abs(neg_mean) if pd.notna(pos_mean) and pd.notna(neg_mean) and neg_mean != 0 else np.nan

    return {
        "ann_ret": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
        "max_drawdown": float(max_dd),
        "calmar": float(calmar) if pd.notna(calmar) else np.nan,
        "win_rate": float(win_rate) if pd.notna(win_rate) else np.nan,
        "profit_loss_ratio": float(pl_ratio) if pd.notna(pl_ratio) else np.nan,
    }


def calc_split_metrics(ret: pd.Series, split_date: str) -> List[Dict[str, float]]:
    """按样本外起点拆分收益，并分别计算绩效指标。"""
    split_ts = pd.Timestamp(split_date)
    clean_ret = ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    parts = [
        ("in_sample", clean_ret[clean_ret.index < split_ts]),
        ("out_of_sample", clean_ret[clean_ret.index >= split_ts]),
    ]

    records = []
    for period, part_ret in parts:
        if part_ret.empty:
            continue
        nav = (1.0 + part_ret).cumprod()
        m = calc_metrics(part_ret, nav)
        records.append({
            "period": period,
            "start": part_ret.index.min().date().isoformat(),
            "end": part_ret.index.max().date().isoformat(),
            "n_days": int(len(part_ret)),
            **m,
            "final_nav": float(nav.iloc[-1]),
        })
    return records


def generate_position_state_machine(
    price: pd.Series,
    hurst: pd.Series,
    short_win: int,
    long_win: int,
    dev_win: int,
    bb_k: float = DEFAULT_BB_K,
    stop_daily_ret: float = DEFAULT_STOP_DAILY_RET,
    max_hold: int = DEFAULT_MAX_HOLD,
) -> pd.Series:
    """
    生成日频目标仓位：1 多，-1 空，0 空仓。

    约定：
    - 第 t 日收盘后根据信号调整 position[t]；
    - 默认回测口径为下一交易日开盘调仓；
    - position[t] 参与 open[t+1] 到 open[t+2] 的收益。
    """
    price = price.astype(float)
    ret = price.pct_change().replace([np.inf, -np.inf], np.nan)

    h_dev = hurst - hurst.rolling(dev_win, min_periods=max(5, dev_win // 2)).mean()

    ma_s = price.rolling(short_win, min_periods=short_win).mean()
    ma_l = price.rolling(long_win, min_periods=long_win).mean()

    bb_mid = price.rolling(short_win, min_periods=short_win).mean()
    bb_std = price.rolling(short_win, min_periods=short_win).std(ddof=1)
    bb_upper = bb_mid + bb_k * bb_std
    bb_lower = bb_mid - bb_k * bb_std

    idx = price.index
    pos = pd.Series(0.0, index=idx, name="position")
    current_pos = 0.0
    hold_days = 0

    for i in range(1, len(idx)):
        # 先假设收盘后延续当前仓位
        next_pos = current_pos

        # 如果已经有仓位：只检查平仓，不做反手
        if current_pos != 0:
            hold_days += 1
            daily_move = ret.iloc[i]
            exit_by_daily_move = pd.notna(daily_move) and abs(daily_move) >= stop_daily_ret
            exit_by_max_hold = hold_days >= max_hold
            if exit_by_daily_move or exit_by_max_hold:
                next_pos = 0.0
                hold_days = 0
        else:
            # 空仓时才根据 Hurst 偏离选择一种趋势信号建仓
            sig = 0.0
            if pd.notna(h_dev.iloc[i]):
                if h_dev.iloc[i] > 0:
                    # Hurst 正偏：布林带通道突破，作为极端趋势信号
                    long_sig = (
                        pd.notna(bb_upper.iloc[i]) and pd.notna(bb_upper.iloc[i - 1])
                        and price.iloc[i] > bb_upper.iloc[i]
                        and price.iloc[i - 1] <= bb_upper.iloc[i - 1]
                    )
                    short_sig = (
                        pd.notna(bb_lower.iloc[i]) and pd.notna(bb_lower.iloc[i - 1])
                        and price.iloc[i] < bb_lower.iloc[i]
                        and price.iloc[i - 1] >= bb_lower.iloc[i - 1]
                    )
                    if long_sig:
                        sig = 1.0
                    elif short_sig:
                        sig = -1.0
                elif h_dev.iloc[i] < 0:
                    # Hurst 负偏：双均线趋势跟踪
                    long_sig = (
                        pd.notna(ma_s.iloc[i]) and pd.notna(ma_l.iloc[i])
                        and pd.notna(ma_s.iloc[i - 1]) and pd.notna(ma_l.iloc[i - 1])
                        and ma_s.iloc[i] > ma_l.iloc[i]
                        and ma_s.iloc[i - 1] <= ma_l.iloc[i - 1]
                    )
                    short_sig = (
                        pd.notna(ma_s.iloc[i]) and pd.notna(ma_l.iloc[i])
                        and pd.notna(ma_s.iloc[i - 1]) and pd.notna(ma_l.iloc[i - 1])
                        and ma_s.iloc[i] < ma_l.iloc[i]
                        and ma_s.iloc[i - 1] >= ma_l.iloc[i - 1]
                    )
                    if long_sig:
                        sig = 1.0
                    elif short_sig:
                        sig = -1.0

            if sig != 0:
                next_pos = sig
                hold_days = 0

        current_pos = next_pos
        pos.iloc[i] = current_pos

    return pos


def calc_strategy_return(
    position: pd.Series,
    trade_source: pd.DataFrame,
    trade_price_col: str,
    cost: float,
    return_mode: str,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """根据执行口径计算策略收益，并返回对齐后的仓位、收益和换手。"""
    if trade_price_col not in trade_source.columns:
        raise ValueError(f"交易行情中没有 {trade_price_col} 字段，可用字段：{list(trade_source.columns)}")

    trade_price = trade_source[trade_price_col].astype(float).dropna()
    common_index = position.index.intersection(trade_price.index)
    position = position.reindex(common_index).ffill().fillna(0.0)
    trade_price = trade_price.reindex(common_index)
    price_ret = trade_price.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if return_mode == "open_to_open":
        # position[t] 在 t+1 日开盘执行，并吃 open[t+1] 到 open[t+2] 的收益。
        gross_ret = position.shift(2).fillna(0.0) * price_ret
        turnover = position.diff().abs().fillna(position.abs()).shift(1).fillna(0.0)
    elif return_mode == "close_to_close":
        gross_ret = position.shift(1).fillna(0.0) * price_ret
        turnover = position.diff().abs().fillna(position.abs())
    else:
        raise ValueError(f"未知 return_mode: {return_mode}")

    strategy_ret = gross_ret - cost * turnover
    strategy_ret.name = "strategy_ret"
    return position, strategy_ret, turnover


def run_strategy(
    symbol: str,
    df: pd.DataFrame,
    hurst_cache: Dict[Tuple[int, str], pd.Series],
    short_win: int,
    long_win: int,
    dev_win: int,
    hurst_config: str = "rs_default",
    price_col: str = "close",
    trade_df: Optional[pd.DataFrame] = None,
    trade_price_col: Optional[str] = None,
    return_mode: str = "open_to_open",
    cost: float = DEFAULT_COST,
    bb_k: float = DEFAULT_BB_K,
    stop_daily_ret: float = DEFAULT_STOP_DAILY_RET,
    max_hold: int = DEFAULT_MAX_HOLD,
) -> BacktestResult:
    if price_col not in df.columns:
        raise ValueError(f"{symbol} 行情中没有 {price_col} 字段，可用字段：{list(df.columns)}")

    price = df[price_col].astype(float).dropna()
    hurst = hurst_cache[(long_win, hurst_config)].reindex(price.index)

    position = generate_position_state_machine(
        price=price,
        hurst=hurst,
        short_win=short_win,
        long_win=long_win,
        dev_win=dev_win,
        bb_k=bb_k,
        stop_daily_ret=stop_daily_ret,
        max_hold=max_hold,
    )

    trade_source = trade_df if trade_df is not None else df
    trade_col = trade_price_col or price_col
    try:
        position, strategy_ret, turnover = calc_strategy_return(
            position=position,
            trade_source=trade_source,
            trade_price_col=trade_col,
            cost=cost,
            return_mode=return_mode,
        )
    except ValueError as e:
        raise ValueError(f"{symbol} {e}") from e
    nav = (1.0 + strategy_ret.fillna(0.0)).cumprod()
    metrics = calc_metrics(strategy_ret, nav)
    metrics["turnover"] = float(turnover.mean() * TRADING_DAYS)
    metrics["final_nav"] = float(nav.iloc[-1]) if len(nav) else np.nan
    metrics["active_ratio"] = float((position.abs() > 0).mean()) if len(position) else np.nan

    return BacktestResult(
        symbol=symbol,
        short_win=short_win,
        long_win=long_win,
        dev_win=dev_win,
        nav=nav,
        ret=strategy_ret,
        position=position,
        metrics=metrics,
        param_label=f"S={short_win}, L={long_win}, D={dev_win}, H={hurst_config}",
        hurst_config=hurst_config,
    )


def grid_search_symbol(
    symbol: str,
    df: pd.DataFrame,
    trade_df: Optional[pd.DataFrame],
    short_windows: Iterable[int],
    long_windows: Iterable[int],
    dev_windows: Iterable[int],
    hurst_configs: Iterable[str],
    price_col: str,
    trade_price_col: str,
    return_mode: str,
    cost: float,
    bb_k: float,
    stop_daily_ret: float,
    max_hold: int,
) -> Tuple[pd.DataFrame, BacktestResult, Dict[Tuple[int, int, int, str], BacktestResult]]:
    print(f"\n================ {symbol} 参数网格搜索 ================")
    price = df[price_col].astype(float).dropna()
    hurst_cache = precompute_hurst(price, long_windows, hurst_configs)

    records = []
    result_map: Dict[Tuple[int, int, int, str], BacktestResult] = {}

    combos = list(itertools.product(short_windows, long_windows, dev_windows, hurst_configs))
    combos = [(s, l, d, h) for s, l, d, h in combos if s < l]

    for k, (s, l, d, h) in enumerate(combos, start=1):
        bt = run_strategy(
            symbol=symbol,
            df=df,
            hurst_cache=hurst_cache,
            short_win=s,
            long_win=l,
            dev_win=d,
            hurst_config=h,
            price_col=price_col,
            trade_df=trade_df,
            trade_price_col=trade_price_col,
            return_mode=return_mode,
            cost=cost,
            bb_k=bb_k,
            stop_daily_ret=stop_daily_ret,
            max_hold=max_hold,
        )
        result_map[(s, l, d, h)] = bt
        rec = {
            "symbol": symbol,
            "short_win": s,
            "long_win": l,
            "dev_win": d,
            "hurst_config": h,
            "signal_price_col": price_col,
            "trade_price_col": trade_price_col,
            "return_mode": return_mode,
        }
        rec.update(HURST_CONFIGS[h])
        rec.update(bt.metrics)
        records.append(rec)
        if k % 20 == 0 or k == len(combos):
            print(f"  已完成 {k}/{len(combos)}")

    grid = pd.DataFrame(records)
    grid = grid.sort_values(["sharpe", "calmar", "ann_ret"], ascending=[False, False, False])
    best_row = grid.iloc[0]
    best_key = (
        int(best_row["short_win"]),
        int(best_row["long_win"]),
        int(best_row["dev_win"]),
        str(best_row["hurst_config"]),
    )
    best_bt = result_map[best_key]

    print(
        f"✅ {symbol} 最优参数：short={best_key[0]}, long={best_key[1]}, "
        f"dev={best_key[2]}, hurst={best_key[3]} | "
        f"Sharpe={best_bt.metrics['sharpe']:.3f}, AnnRet={best_bt.metrics['ann_ret']:.2%}, "
        f"MaxDD={best_bt.metrics['max_drawdown']:.2%}"
    )
    return grid, best_bt, result_map


def build_parameter_ensemble_result(
    symbol: str,
    df: pd.DataFrame,
    trade_df: Optional[pd.DataFrame],
    result_map: Dict[Tuple[int, int, int, str], BacktestResult],
    price_col: str,
    trade_price_col: str,
    return_mode: str,
    cost: float,
) -> BacktestResult:
    """把多组参数生成的目标仓位做等权平均，形成参数组合策略。"""
    if not result_map:
        raise ValueError(f"{symbol} 没有可用于 ensemble 的参数结果。")
    if price_col not in df.columns:
        raise ValueError(f"{symbol} 行情中没有 {price_col} 字段，可用字段：{list(df.columns)}")

    price = df[price_col].astype(float).dropna()
    pos_dict = {
        f"S{s}_L{l}_D{d}_{h}": bt.position.reindex(price.index)
        for (s, l, d, h), bt in result_map.items()
    }
    position = pd.DataFrame(pos_dict).mean(axis=1, skipna=True).fillna(0.0)
    position.name = "position"

    trade_source = trade_df if trade_df is not None else df
    try:
        position, strategy_ret, turnover = calc_strategy_return(
            position=position,
            trade_source=trade_source,
            trade_price_col=trade_price_col,
            cost=cost,
            return_mode=return_mode,
        )
    except ValueError as e:
        raise ValueError(f"{symbol} {e}") from e
    nav = (1.0 + strategy_ret.fillna(0.0)).cumprod()

    metrics = calc_metrics(strategy_ret, nav)
    metrics["turnover"] = float(turnover.mean() * TRADING_DAYS)
    metrics["final_nav"] = float(nav.iloc[-1]) if len(nav) else np.nan
    metrics["active_ratio"] = float((position.abs() > 1e-12).mean()) if len(position) else np.nan
    metrics["param_count"] = float(len(result_map))

    return BacktestResult(
        symbol=symbol,
        short_win=0,
        long_win=0,
        dev_win=0,
        nav=nav,
        ret=strategy_ret,
        position=position,
        metrics=metrics,
        param_label=f"ensemble mean signal ({len(result_map)} params)",
        hurst_config="ensemble",
    )


# =========================
# 组合层：等权 / 波动率倒数
# =========================
def combine_equal_weight(ret_dict: Dict[str, pd.Series]) -> pd.Series:
    ret_df = pd.DataFrame(ret_dict).sort_index()
    # 每天对有值的品种等权，缺失不参与
    combined = ret_df.mean(axis=1, skipna=True).fillna(0.0)
    combined.name = "equal_weight_ret"
    return combined


def combine_inverse_vol(ret_dict: Dict[str, pd.Series], vol_window: int = 120) -> Tuple[pd.Series, pd.DataFrame]:
    ret_df = pd.DataFrame(ret_dict).sort_index()
    rolling_vol = ret_df.rolling(vol_window, min_periods=max(20, vol_window // 3)).std(ddof=1)
    inv_vol = 1.0 / rolling_vol.replace(0, np.nan)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    valid_ret = ret_df.notna()
    tradable_weights = weights.shift(1).fillna(0.0).where(valid_ret, 0.0)
    weight_sum = tradable_weights.sum(axis=1).replace(0, np.nan)
    tradable_weights = tradable_weights.div(weight_sum, axis=0).fillna(0.0)
    combined = (tradable_weights * ret_df.fillna(0.0)).sum(axis=1)
    combined.name = "inverse_vol_ret"
    return combined, tradable_weights


def _prefixed_metrics(prefix: str, ret: pd.Series) -> Dict[str, float]:
    """计算组合指标，并给字段加前缀，便于同一行比较不同组合方式。"""
    nav = (1.0 + ret.fillna(0.0)).cumprod()
    metrics = calc_metrics(ret, nav)
    result = {f"{prefix}_{k}": v for k, v in metrics.items()}
    result[f"{prefix}_final_nav"] = float(nav.iloc[-1]) if len(nav) else np.nan
    return result


def build_global_fixed_parameter_grid(
    result_maps: Dict[str, Dict[Tuple[int, int, int, str], BacktestResult]],
    vol_window: int = 120,
) -> pd.DataFrame:
    """
    用同一组参数横跨所有品种构建组合，评估“全局固定参数”的样本内表现。

    这张表用于从训练期选择一组统一固定参数；它不参与单品种 best 的选择，
    也不会改变 best / ensemble / fixed 模式已有输出。
    """
    if not result_maps:
        return pd.DataFrame()

    common_keys = set.intersection(*(set(m.keys()) for m in result_maps.values() if m))
    records = []

    for short_win, long_win, dev_win, hurst_config in sorted(common_keys):
        ret_dict = {
            symbol: symbol_map[(short_win, long_win, dev_win, hurst_config)].ret
            for symbol, symbol_map in result_maps.items()
            if (short_win, long_win, dev_win, hurst_config) in symbol_map
        }
        if not ret_dict:
            continue

        eq_ret = combine_equal_weight(ret_dict)
        invvol_ret, _ = combine_inverse_vol(ret_dict, vol_window=vol_window)

        record = {
            "short_win": short_win,
            "long_win": long_win,
            "dev_win": dev_win,
            "hurst_config": hurst_config,
            "n_symbols": len(ret_dict),
        }
        record.update(HURST_CONFIGS[hurst_config])
        record.update(_prefixed_metrics("equal", eq_ret))
        record.update(_prefixed_metrics("invvol", invvol_ret))
        records.append(record)

    if not records:
        return pd.DataFrame()

    grid = pd.DataFrame(records)
    return grid.sort_values(
        ["invvol_calmar", "invvol_sharpe", "equal_calmar", "equal_sharpe"],
        ascending=[False, False, False, False],
    )


# =========================
# 输出图表
# =========================
def ensure_matplotlib():
    import matplotlib

    # 服务器/无界面环境下也能保存图
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_best_nav(symbol: str, df: pd.DataFrame, bt: BacktestResult, price_col: str, out_path: Path) -> None:
    plt = ensure_matplotlib()
    price = df[price_col].reindex(bt.nav.index).astype(float)
    price_nav = price / price.dropna().iloc[0]
    drawdown = bt.nav / bt.nav.cummax() - 1.0
    param_text = bt.param_label or f"S={bt.short_win}, L={bt.long_win}, D={bt.dev_win}"

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(price_nav.index, price_nav.values, label="underlying normalized")
    ax1.plot(bt.nav.index, bt.nav.values, label="strategy nav")
    ax1.set_title(
        f"{symbol} Hurst Trend Enhancement | "
        f"{param_text}, "
        f"Sharpe={bt.metrics['sharpe']:.2f}, MaxDD={bt.metrics['max_drawdown']:.2%}"
    )
    ax1.set_ylabel("NAV")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.15, label="drawdown")
    ax2.set_ylabel("Drawdown")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_sensitivity_3d(symbol: str, grid: pd.DataFrame, out_path: Path, metric: str = "sharpe") -> None:
    plt = ensure_matplotlib()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    g = grid.dropna(subset=[metric]).copy()
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(
        g["short_win"],
        g["long_win"],
        g["dev_win"],
        c=g[metric],
        s=55,
        alpha=0.85,
    )
    ax.set_xlabel("short / bollinger window")
    ax.set_ylabel("long / hurst window")
    ax.set_zlabel("hurst deviation window")
    ax.set_title(f"{symbol} parameter sensitivity by {metric}")
    fig.colorbar(p, ax=ax, shrink=0.65, label=metric)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_sensitivity_heatmap(symbol: str, grid: pd.DataFrame, out_path: Path, metric: str = "sharpe") -> None:
    plt = ensure_matplotlib()
    devs = sorted(grid["dev_win"].unique())
    configs = sorted(grid["hurst_config"].unique()) if "hurst_config" in grid.columns else [None]
    panels = list(itertools.product(configs, devs))
    n_cols = len(devs)
    n_rows = len(configs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.7 * n_rows), sharey=True)
    axes = np.asarray(axes).reshape(n_rows, n_cols)
    im = None

    for ax, (cfg, d) in zip(axes.ravel(), panels):
        sub = grid[grid["dev_win"] == d]
        if cfg is not None:
            sub = sub[sub["hurst_config"] == cfg]
        pivot = sub.pivot_table(index="long_win", columns="short_win", values=metric, aggfunc="mean")
        im = ax.imshow(pivot.values, aspect="auto", origin="lower")
        title = f"dev={d}" if cfg is None else f"{cfg}, dev={d}"
        ax.set_title(title)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("short")
        ax.set_ylabel("long")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)
    fig.suptitle(f"{symbol} sensitivity heatmap by {metric}")
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label=metric)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_portfolio_nav(ret: pd.Series, name: str, out_path: Path) -> None:
    plt = ensure_matplotlib()
    nav = (1.0 + ret.fillna(0.0)).cumprod()
    metrics = calc_metrics(ret, nav)
    dd = nav / nav.cummax() - 1.0

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(nav.index, nav.values, label=name)
    ax1.set_title(
        f"{name} | AnnRet={metrics['ann_ret']:.2%}, Vol={metrics['ann_vol']:.2%}, "
        f"Sharpe={metrics['sharpe']:.2f}, MaxDD={metrics['max_drawdown']:.2%}, Calmar={metrics['calmar']:.2f}"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.fill_between(dd.index, dd.values, 0, alpha=0.15)
    ax2.set_ylabel("Drawdown")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# =========================
# 主程序
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="复现 Hurst 趋势增强 CTA 日频时序策略")
    parser.add_argument("--start", default="2014-01-01", help="开始日期")
    parser.add_argument("--end", default="2024-06-06", help="结束日期")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS, help="期货品种代码，例如 CU AL ZN NI")
    parser.add_argument("--output", default="hurst_cta_outputs", help="输出目录")
    parser.add_argument(
        "--strategy-mode",
        default="best",
        choices=["best", "ensemble", "fixed"],
        help="组合层使用的单品种策略：best=样本内最优参数；ensemble=多参数仓位平均；fixed=固定参数",
    )
    parser.add_argument("--short-windows", nargs="+", type=int, default=SHORT_WINDOWS, help="参与网格搜索/ensemble 的 short_win 参数族")
    parser.add_argument("--long-windows", nargs="+", type=int, default=LONG_WINDOWS, help="参与网格搜索/ensemble 的 long_win 参数族")
    parser.add_argument("--dev-windows", nargs="+", type=int, default=DEV_WINDOWS, help="参与网格搜索/ensemble 的 dev_win 参数族")
    parser.add_argument("--fixed-short", type=int, default=60, help="fixed 模式的 short_win")
    parser.add_argument("--fixed-long", type=int, default=140, help="fixed 模式的 long_win")
    parser.add_argument("--fixed-dev", type=int, default=60, help="fixed 模式的 dev_win")
    parser.add_argument("--fixed-hurst-config", default="rs_default", choices=list(HURST_CONFIGS), help="fixed 模式的 Hurst R/S 估计口径")
    parser.add_argument(
        "--hurst-configs",
        nargs="+",
        default=DEFAULT_HURST_CONFIGS,
        choices=list(HURST_CONFIGS),
        help="参与网格搜索的 Hurst R/S 估计口径，可传入 rs_fast rs_default rs_robust",
    )
    parser.add_argument("--oos-start", default=None, help="样本外开始日期，例如 2024-06-07；为空则不输出样本内/样本外拆分")
    parser.add_argument("--price-col", default="close", choices=["close", "settlement"], help="用于计算信号的价格字段")
    parser.add_argument("--trade-price-col", default="open", choices=["open", "close", "settlement"], help="用于计算收益的价格字段，默认 open")
    parser.add_argument(
        "--return-mode",
        default="open_to_open",
        choices=["open_to_open", "close_to_close"],
        help="收益计算口径：open_to_open=收盘算信号、下一日开盘调仓；close_to_close=旧版收盘到收盘口径",
    )
    parser.add_argument("--adjust-type", default="pre", choices=["pre", "post", "none"], help="信号价格的主力连续复权方式")
    parser.add_argument("--adjust-method", default="prev_close_ratio", help="信号价格的主力连续平滑方法")
    parser.add_argument("--trade-adjust-type", default="none", choices=["pre", "post", "none"], help="收益价格的主力连续复权方式，建议 none")
    parser.add_argument("--trade-adjust-method", default=None, help="收益价格的主力连续平滑方法，默认与 --adjust-method 相同")
    parser.add_argument("--cost", type=float, default=DEFAULT_COST, help="单边交易成本")
    parser.add_argument("--bb-k", type=float, default=DEFAULT_BB_K, help="布林带标准差倍数")
    parser.add_argument("--stop-daily-ret", type=float, default=DEFAULT_STOP_DAILY_RET, help="单日涨跌幅止盈止损阈值")
    parser.add_argument("--max-hold", type=int, default=DEFAULT_MAX_HOLD, help="最长持有期")
    parser.add_argument("--username", default=None, help="RQDatac username；建议改用环境变量")
    parser.add_argument("--password", default=None, help="RQDatac password；建议改用环境变量")
    parser.add_argument("--force-download", action="store_true", help="强制重新下载行情，不使用缓存")
    parser.add_argument("--skip-rq-init", action="store_true", help="如果只用本地缓存数据，可跳过 rqdatac 初始化")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output)
    data_dir = out_dir / "data"
    result_dir = out_dir / "results"
    plot_dir = out_dir / "plots"
    position_dir = out_dir / "positions"
    for p in [data_dir, result_dir, plot_dir, position_dir]:
        p.mkdir(parents=True, exist_ok=True)

    if not args.skip_rq_init:
        init_rqdatac(args.username, args.password)

    trade_price_col = args.trade_price_col or ("open" if args.return_mode == "open_to_open" else args.price_col)
    trade_adjust_method = args.trade_adjust_method or args.adjust_method
    short_windows = sorted(dict.fromkeys(args.short_windows))
    long_windows = sorted(dict.fromkeys(args.long_windows))
    dev_windows = sorted(dict.fromkeys(args.dev_windows))
    hurst_configs = list(dict.fromkeys(args.hurst_configs))
    if args.strategy_mode == "fixed":
        if args.fixed_short not in short_windows:
            short_windows.append(args.fixed_short)
            short_windows = sorted(short_windows)
        if args.fixed_long not in long_windows:
            long_windows.append(args.fixed_long)
            long_windows = sorted(long_windows)
        if args.fixed_dev not in dev_windows:
            dev_windows.append(args.fixed_dev)
            dev_windows = sorted(dev_windows)
    if args.strategy_mode == "fixed" and args.fixed_hurst_config not in hurst_configs:
        hurst_configs.append(args.fixed_hurst_config)
    print(
        "▶ 参数族："
        f"short={short_windows}, long={long_windows}, dev={dev_windows}, "
        f"hurst={hurst_configs}, mode={args.strategy_mode}"
    )

    data = load_all_data(
        symbols=args.symbols,
        start_date=args.start,
        end_date=args.end,
        data_dir=data_dir,
        adjust_type=args.adjust_type,
        adjust_method=args.adjust_method,
        force_download=args.force_download,
    )
    if not data:
        raise RuntimeError("没有任何品种取到数据，程序结束。")

    if args.trade_adjust_type == args.adjust_type and trade_adjust_method == args.adjust_method:
        trade_data = data
    else:
        trade_data = load_all_data(
            symbols=data.keys(),
            start_date=args.start,
            end_date=args.end,
            data_dir=data_dir,
            adjust_type=args.trade_adjust_type,
            adjust_method=trade_adjust_method,
            force_download=args.force_download,
        )

    best_records = []
    strategy_records = []
    strategy_period_records = []
    strategy_ret_dict: Dict[str, pd.Series] = {}
    all_grid_list = []
    all_result_maps: Dict[str, Dict[Tuple[int, int, int, str], BacktestResult]] = {}

    for symbol, df in data.items():
        if args.price_col not in df.columns or df[args.price_col].dropna().empty:
            print(f"⚠️ {symbol} 缺少 {args.price_col}，跳过")
            continue
        trade_df = trade_data.get(symbol)
        if trade_df is None or trade_price_col not in trade_df.columns or trade_df[trade_price_col].dropna().empty:
            print(f"⚠️ {symbol} 缺少交易价格 {trade_price_col}，跳过")
            continue

        grid, best_bt, result_map = grid_search_symbol(
            symbol=symbol,
            df=df,
            trade_df=trade_df,
            short_windows=short_windows,
            long_windows=long_windows,
            dev_windows=dev_windows,
            hurst_configs=hurst_configs,
            price_col=args.price_col,
            trade_price_col=trade_price_col,
            return_mode=args.return_mode,
            cost=args.cost,
            bb_k=args.bb_k,
            stop_daily_ret=args.stop_daily_ret,
            max_hold=args.max_hold,
        )
        all_result_maps[symbol] = result_map

        if args.strategy_mode == "best":
            strategy_bt = best_bt
        elif args.strategy_mode == "ensemble":
            strategy_bt = build_parameter_ensemble_result(
                symbol=symbol,
                df=df,
                trade_df=trade_df,
                result_map=result_map,
                price_col=args.price_col,
                trade_price_col=trade_price_col,
                return_mode=args.return_mode,
                cost=args.cost,
            )
        else:
            fixed_key = (args.fixed_short, args.fixed_long, args.fixed_dev, args.fixed_hurst_config)
            if fixed_key not in result_map:
                raise ValueError(
                    f"{symbol} fixed 参数不在网格中：short={args.fixed_short}, "
                    f"long={args.fixed_long}, dev={args.fixed_dev}, "
                    f"hurst_config={args.fixed_hurst_config}"
                )
            strategy_bt = result_map[fixed_key]

        grid_path = result_dir / f"{symbol}_parameter_grid.csv"
        grid.to_csv(grid_path, index=False, encoding="utf-8-sig")
        all_grid_list.append(grid)

        nav_path = plot_dir / f"{symbol}_best_nav.png"
        scatter_path = plot_dir / f"{symbol}_sensitivity_3d.png"
        heatmap_path = plot_dir / f"{symbol}_sensitivity_heatmap.png"
        plot_best_nav(symbol, df, best_bt, args.price_col, nav_path)
        plot_sensitivity_3d(symbol, grid, scatter_path, metric="sharpe")
        plot_sensitivity_heatmap(symbol, grid, heatmap_path, metric="sharpe")

        best_pos_df = pd.DataFrame({
            "position": best_bt.position,
            "strategy_ret": best_bt.ret,
            "strategy_nav": best_bt.nav,
        })
        best_pos_df.to_csv(position_dir / f"{symbol}_best_position_return.csv", encoding="utf-8-sig")

        if args.strategy_mode != "best":
            plot_best_nav(symbol, df, strategy_bt, args.price_col, plot_dir / f"{symbol}_{args.strategy_mode}_nav.png")
            strategy_pos_df = pd.DataFrame({
                "position": strategy_bt.position,
                "strategy_ret": strategy_bt.ret,
                "strategy_nav": strategy_bt.nav,
            })
            strategy_pos_df.to_csv(
                position_dir / f"{symbol}_{args.strategy_mode}_position_return.csv",
                encoding="utf-8-sig",
            )

        best_rec = {
            "symbol": symbol,
            "short_win": best_bt.short_win,
            "long_win": best_bt.long_win,
            "dev_win": best_bt.dev_win,
            "hurst_config": best_bt.hurst_config,
            "signal_price_col": args.price_col,
            "trade_price_col": trade_price_col,
            "return_mode": args.return_mode,
        }
        best_rec.update(best_bt.metrics)
        best_records.append(best_rec)

        strategy_rec = {
            "symbol": symbol,
            "strategy_mode": args.strategy_mode,
            "param_label": strategy_bt.param_label,
            "short_win": strategy_bt.short_win if args.strategy_mode in ["best", "fixed"] else np.nan,
            "long_win": strategy_bt.long_win if args.strategy_mode in ["best", "fixed"] else np.nan,
            "dev_win": strategy_bt.dev_win if args.strategy_mode in ["best", "fixed"] else np.nan,
            "hurst_config": strategy_bt.hurst_config if args.strategy_mode in ["best", "fixed"] else "ensemble",
            "signal_price_col": args.price_col,
            "trade_price_col": trade_price_col,
            "return_mode": args.return_mode,
        }
        strategy_rec.update(strategy_bt.metrics)
        strategy_records.append(strategy_rec)
        strategy_ret_dict[symbol] = strategy_bt.ret

        if args.oos_start:
            for split_rec in calc_split_metrics(strategy_bt.ret, args.oos_start):
                strategy_period_records.append({
                    "symbol": symbol,
                    "strategy_mode": args.strategy_mode,
                    "param_label": strategy_bt.param_label,
                    "signal_price_col": args.price_col,
                    "trade_price_col": trade_price_col,
                    "return_mode": args.return_mode,
                    **split_rec,
                })

    if not strategy_records:
        raise RuntimeError("没有完成任何品种的回测，请检查数据。")

    best_summary = pd.DataFrame(best_records).sort_values("sharpe", ascending=False)
    best_summary.to_csv(result_dir / "best_summary_by_symbol.csv", index=False, encoding="utf-8-sig")
    print("\n================ 单品种最优参数汇总 ================")
    print(best_summary.to_string(index=False))

    strategy_summary = pd.DataFrame(strategy_records).sort_values("sharpe", ascending=False)
    if args.strategy_mode != "best":
        strategy_summary.to_csv(
            result_dir / f"{args.strategy_mode}_summary_by_symbol.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(f"\n================ 单品种 {args.strategy_mode} 策略汇总 ================")
        print(strategy_summary.to_string(index=False))

    if strategy_period_records:
        pd.DataFrame(strategy_period_records).to_csv(
            result_dir / f"{args.strategy_mode}_period_summary_by_symbol.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if all_grid_list:
        all_grid = pd.concat(all_grid_list, ignore_index=True)
        all_grid.to_csv(result_dir / "all_symbols_parameter_grid.csv", index=False, encoding="utf-8-sig")

    global_fixed_grid = build_global_fixed_parameter_grid(all_result_maps, vol_window=120)
    if not global_fixed_grid.empty:
        global_fixed_grid.insert(0, "signal_price_col", args.price_col)
        global_fixed_grid.insert(1, "trade_price_col", trade_price_col)
        global_fixed_grid.insert(2, "return_mode", args.return_mode)
        global_fixed_grid.to_csv(
            result_dir / "global_fixed_parameter_grid.csv",
            index=False,
            encoding="utf-8-sig",
        )
        global_fixed_grid.head(20).to_csv(
            result_dir / "global_fixed_top20.csv",
            index=False,
            encoding="utf-8-sig",
        )
        best_global_fixed = global_fixed_grid.iloc[0]
        pd.DataFrame([best_global_fixed]).to_csv(
            result_dir / "global_fixed_best_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print("\n================ 全局固定参数 Top 1 ================")
        print(
            "short={short}, long={long}, dev={dev}, hurst={hurst} | "
            "InvVol Calmar={calmar:.3f}, InvVol Sharpe={sharpe:.3f}, "
            "Equal Calmar={eq_calmar:.3f}, Equal Sharpe={eq_sharpe:.3f}".format(
                short=int(best_global_fixed["short_win"]),
                long=int(best_global_fixed["long_win"]),
                dev=int(best_global_fixed["dev_win"]),
                hurst=str(best_global_fixed["hurst_config"]),
                calmar=float(best_global_fixed["invvol_calmar"]),
                sharpe=float(best_global_fixed["invvol_sharpe"]),
                eq_calmar=float(best_global_fixed["equal_calmar"]),
                eq_sharpe=float(best_global_fixed["equal_sharpe"]),
            )
        )

    # 组合：全品种等权、全品种定长波动率倒数加权
    eq_ret = combine_equal_weight(strategy_ret_dict)
    invvol_ret, invvol_w = combine_inverse_vol(strategy_ret_dict, vol_window=120)

    portfolio_summary = []
    portfolio_period_summary = []
    for name, ret in [("all_equal_weight", eq_ret), ("all_inverse_vol_120d", invvol_ret)]:
        nav = (1.0 + ret.fillna(0.0)).cumprod()
        m = calc_metrics(ret, nav)
        portfolio_summary.append({
            "portfolio": name,
            "group": "all",
            "weight_method": "equal_weight" if name.endswith("equal_weight") else "inverse_vol_120d",
            "strategy_mode": args.strategy_mode,
            "signal_price_col": args.price_col,
            "trade_price_col": trade_price_col,
            "return_mode": args.return_mode,
            **m,
            "final_nav": float(nav.iloc[-1]),
        })
        if args.oos_start:
            for split_rec in calc_split_metrics(ret, args.oos_start):
                portfolio_period_summary.append({
                    "portfolio": name,
                    "group": "all",
                    "weight_method": "equal_weight" if name.endswith("equal_weight") else "inverse_vol_120d",
                    "strategy_mode": args.strategy_mode,
                    "signal_price_col": args.price_col,
                    "trade_price_col": trade_price_col,
                    "return_mode": args.return_mode,
                    **split_rec,
                })
        pd.DataFrame({"ret": ret, "nav": nav}).to_csv(result_dir / f"{name}_return_nav.csv", encoding="utf-8-sig")
        plot_portfolio_nav(ret, name, plot_dir / f"{name}_nav.png")

    invvol_w.to_csv(result_dir / "all_inverse_vol_120d_weights.csv", encoding="utf-8-sig")

    # 分板块组合
    for group_name, members in GROUPS.items():
        sub = {k: v for k, v in strategy_ret_dict.items() if k in members}
        if len(sub) < 2:
            continue
        group_eq = combine_equal_weight(sub)
        group_inv, group_w = combine_inverse_vol(sub, vol_window=120)
        for name, ret in [(f"{group_name}_equal_weight", group_eq), (f"{group_name}_inverse_vol_120d", group_inv)]:
            nav = (1.0 + ret.fillna(0.0)).cumprod()
            m = calc_metrics(ret, nav)
            portfolio_summary.append({
                "portfolio": name,
                "group": group_name,
                "weight_method": "equal_weight" if name.endswith("equal_weight") else "inverse_vol_120d",
                "strategy_mode": args.strategy_mode,
                "signal_price_col": args.price_col,
                "trade_price_col": trade_price_col,
                "return_mode": args.return_mode,
                **m,
                "final_nav": float(nav.iloc[-1]),
            })
            if args.oos_start:
                for split_rec in calc_split_metrics(ret, args.oos_start):
                    portfolio_period_summary.append({
                        "portfolio": name,
                        "group": group_name,
                        "weight_method": "equal_weight" if name.endswith("equal_weight") else "inverse_vol_120d",
                        "strategy_mode": args.strategy_mode,
                        "signal_price_col": args.price_col,
                        "trade_price_col": trade_price_col,
                        "return_mode": args.return_mode,
                        **split_rec,
                    })
            pd.DataFrame({"ret": ret, "nav": nav}).to_csv(result_dir / f"{name}_return_nav.csv", encoding="utf-8-sig")
            plot_portfolio_nav(ret, name, plot_dir / f"{name}_nav.png")
        group_w.to_csv(result_dir / f"{group_name}_inverse_vol_120d_weights.csv", encoding="utf-8-sig")

    pd.DataFrame(portfolio_summary).to_csv(result_dir / "portfolio_summary.csv", index=False, encoding="utf-8-sig")
    if portfolio_period_summary:
        pd.DataFrame(portfolio_period_summary).to_csv(
            result_dir / f"{args.strategy_mode}_portfolio_period_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print(f"\n✅ 全部完成。输出目录：{out_dir.resolve()}")
    print("主要文件：")
    print(f"- 单品种最优参数：{result_dir / 'best_summary_by_symbol.csv'}")
    print(f"- 所有参数网格：{result_dir / 'all_symbols_parameter_grid.csv'}")
    print(f"- 参数敏感性图：{plot_dir}")
    print(f"- 最优仓位与收益：{position_dir}")


if __name__ == "__main__":
    main()
