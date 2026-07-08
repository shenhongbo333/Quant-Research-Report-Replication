# Hurst CTA 时序策略当前代码与回测结果整理

## 1. 当前代码能做什么

当前脚本为 `hurst_cta_reproduction_recent.py`，用于复现和扩展 Hurst 趋势增强 CTA 日频时序策略。当前版本的核心功能包括：
| 序号 | 功能 | 对应函数 / 模块 | 当前设置或说明 |
|---:|---|---|---|
| 1 | 获取商品期货主力连续合约日频数据 | `fetch_dominant_daily()`、`load_all_data()`；底层调用 `rqdatac.futures.get_dominant_price()` | 按品种拉取主力连续行情，并缓存到 `data/` 目录 |
| 2 | 使用主力连续比例复权价格计算信号 | `fetch_dominant_daily()`、`run_strategy()`、`generate_position_state_machine()` | 信号价格使用 `--adjust-type pre --adjust-method prev_close_ratio`；用于计算 Hurst、均线、布林带 |
| 3 | 使用未复权主力价格计算交易收益 | `fetch_dominant_daily()`、`calc_strategy_return()` | 交易价格使用 `--trade-adjust-type none`；收益价格字段为 `--trade-price-col open` |
| 4 | 支持收盘生成信号、下一交易日开盘调仓 | `calc_strategy_return()` | 当前使用 `--return-mode open_to_open`；`position[t]` 是 t 日收盘后形成的目标仓位，实际吃到的是下一段 open-to-open 收益 |
| 5 | 支持三种策略模式 | `grid_search_symbol()`、`build_parameter_ensemble_result()`、`main()` 中的 `strategy_mode` 分支 | `best`：每品种选择样本内最优参数；`ensemble`：多参数仓位平均；`fixed`：所有品种使用指定固定参数 |
| 6 | 支持 Hurst R/S 内部尺度口径遍历 | `hurst_rs()`、`rolling_hurst()`、`precompute_hurst()`、`HURST_CONFIGS` | `rs_fast={min_chunk:6, n_scales:6, min_segments:2, min_valid_scales:3}`；`rs_default={min_chunk:8, n_scales:8, min_segments:2, min_valid_scales:3}`；`rs_robust={min_chunk:10, n_scales:12, min_segments:3, min_valid_scales:4}` |
| 7 | 支持全局固定参数搜索 | `build_global_fixed_parameter_grid()`、`combine_equal_weight()`、`combine_inverse_vol()`、`calc_metrics()` | 对每一组参数，要求所有品种统一使用同一组参数；分别计算全品种等权和 120 日波动率倒数加权组合；输出 `global_fixed_parameter_grid.csv`、`global_fixed_top20.csv`、`global_fixed_best_summary.csv` |
| 8 | 支持样本内 / 样本外拆分统计 | `calc_split_metrics()` | 通过 `--oos-start 2024-06-07` 将结果拆成 `in_sample` 和 `out_of_sample`；输出 `fixed_period_summary_by_symbol.csv` 和 `fixed_portfolio_period_summary.csv` |

## 2. 当前主要数据与交易口径

当前较合理的回测口径如下：

| 项目 | 当前设置 | 说明 |
|---|---:|---|
| 信号价格 | `--price-col close` | 用收盘价计算 Hurst、均线、布林带和交易信号 |
| 收益价格 | `--trade-price-col open` | 用开盘价计算交易收益 |
| 收益模式 | `--return-mode open_to_open` | 收盘算信号，下一日开盘调仓 |
| 信号价格复权 | `--adjust-type pre --adjust-method prev_close_ratio` | 使用比例复权主力连续价格计算信号 |
| 交易价格复权 | `--trade-adjust-type none` | 使用未复权主力价格计算收益 |
| 交易成本 | 默认 `1e-4` | 单边万分之一 |
| 样本内区间 | `2014-01-01 ~ 2024-06-06` | 用于参数敏感性和固定参数筛选 |
| 样本外区间 | `2024-06-07 ~ 2026-06-16` | 用于固定参数样本外验证 |

## 3. Hurst 参数说明

代码中 Hurst 有两层参数。

第一层是 `long_win`，即滚动计算 Hurst 的历史窗口：

```text
long_win = 120, 140, 160, 180
```

第二层是 R/S 方法内部的尺度 `n`。当前不是逐个手工遍历单一 `n`，而是通过三组 `hurst_config` 控制内部尺度生成方式：

| Hurst 口径 | min_chunk | n_scales | min_segments | 含义 |
|---|---:|---:|---:|---|
| `rs_fast` | 6 | 6 | 2 | 尺度较少，估计更快，但可能更敏感 |
| `rs_default` | 8 | 8 | 2 | 默认口径 |
| `rs_robust` | 10 | 12 | 3 | 尺度更多，约束更强，更偏稳健 |

例如 `long_win=140` 时，三组内部 `n` 大致为：

```text
rs_fast:    [6, 9, 16, 26, 42, 70]
rs_default: [8, 10, 14, 20, 27, 37, 51, 70]
rs_robust:  [10, 11, 14, 17, 20, 24, 28, 34, 41, 49, 58, 70]
```

因此，当前代码已经对 Hurst 内部尺度做了“口径级遍历”，但不是对每个单独 `n` 做完全枚举。

## 4. 回测结果


### 4.1 正式报告结果

建议正式报告以下三组。

| 输出包 | 是否正式报告 | 用途 |
|---|---|---|
| `hurst_cta_train_global_fixed_search.zip` | 是 | 样本内 full grid，全局固定参数搜索 |
| `hurst_cta_oos_fixed_top1.zip` | 是 | 固定参数 Top1 的样本外验证 |
| `hurst_cta_oos_fixed_top2.zip` | 是 | 固定参数 Top2 的样本外验证 |

其中：

```text
Top1 = short=20, long=120, dev=40, hurst=rs_fast
Top2 = short=60, long=140, dev=60, hurst=rs_default
```

### 4.2 作为探索记录报告


| 输出包 | 是否正式报告 | 用途 |
|---|---|---|
| `hurst_cta_train_ensemble_family.zip` | 探索记录 | 较宽参数族的 ensemble 平均信号 |
| `hurst_cta_train_ensemble_family_narrow.zip` | 探索记录 | 缩窄参数族后的 ensemble 平均信号 |



## 5. 样本内参数敏感性与 per-symbol best 结果

`best` 模式下每个品种各自选择最优参数：

| 组合 | 年化收益 | 年化波动 | Sharpe | 最大回撤 | Calmar | 最终净值 |
|---|---:|---:|---:|---:|---:|---:|
| 全品种等权 | 15.69% | 6.75% | 2.33 | 6.12% | 2.56 | 4.33 |
| 全品种波动率倒数 | 17.21% | 7.31% | 2.36 | 5.56% | 3.10 | 4.94 |

但这个结果不能作为最终策略表现，因为它的逻辑是：

```text
每个品种各自从参数网格里选择自己历史最优参数，再合成组合。
```

这存在明显的样本内优化偏差。因此，这组结果主要用于说明：

1. 策略在参数空间中存在有效区域。
2. 不同品种的参数稳定性不同。
3. 后续需要使用固定参数、参数族平均或滚动选参降低过拟合。

从参数稳定性看，样本内中位数 Sharpe 较好的品种主要是：

| 品种 | 参数 Sharpe 中位数 | Sharpe > 0 比例 | Sharpe > 0.5 比例 |
|---|---:|---:|---:|
| ZN | 0.238 | 86.11% | 15.28% |
| CU | 0.142 | 68.40% | 8.68% |
| RB | 0.135 | 69.10% | 15.97% |
| JM | 0.135 | 62.50% | 10.76% |
| AL | 0.014 | 51.04% | 1.39% |

表现较弱的品种包括 RU、V、NI、I、TA 等。

## 6. 全局固定参数搜索结果

需要注意的是，本节结果均为样本内结果，回测区间为 `2014-01-01 ~ 2024-06-06`。
`global_fixed_search` 的核心意义是：

```text
所有品种统一使用同一组参数，观察组合层表现。
```

这比 per-symbol best 更严谨，因为它不允许每个品种单独挑选最优参数。

全局固定参数 Top 5 如下：

| 排名 | short | long | dev | Hurst 口径 | 等权 Sharpe | 等权 Calmar | 波动率倒数 Sharpe | 波动率倒数 Calmar | 波动率倒数最大回撤 |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 20 | 120 | 40 | rs_fast | 0.631 | 0.411 | 0.850 | 0.722 | 8.90% |
| 2 | 60 | 140 | 60 | rs_default | 0.925 | 0.433 | 1.020 | 0.596 | 12.83% |
| 3 | 100 | 140 | 60 | rs_fast | 0.698 | 0.414 | 0.743 | 0.528 | 10.44% |
| 4 | 20 | 140 | 20 | rs_fast | 0.694 | 0.396 | 0.909 | 0.528 | 13.39% |
| 5 | 10 | 120 | 60 | rs_robust | 0.598 | 0.431 | 0.725 | 0.526 | 10.09% |

选择 Top1 和 Top2 做样本外验证的原因：

1. Top1 的波动率倒数 Calmar 最高，回撤控制较好。
2. Top2 的 Sharpe 更高，且 `long=140, dev=60, rs_default` 更接近之前观察到的稳健区域。

## 7. Ensemble 参数族平均信号结果

导师建议过“多组参数平均信号”。因此做过两组 ensemble 尝试。

### 7.1 较宽参数族 ensemble

参数族大致为：

```text
short = 20, 40, 60, 80, 100
long = 140
dev = 20, 40, 60, 80
hurst = rs_fast, rs_default, rs_robust
```

组合结果：

| 组合 | 年化收益 | Sharpe | 最大回撤 | Calmar | 最终净值 |
|---|---:|---:|---:|---:|---:|
| 全品种等权 | 3.13% | 0.641 | 10.28% | 0.305 | 1.36 |
| 全品种波动率倒数 | 3.59% | 0.797 | 7.97% | 0.450 | 1.43 |

### 7.2 缩窄参数族 ensemble

参数族大致为：

```text
short = 20, 40, 60, 80
long = 140
dev = 20, 40, 60
hurst = rs_default, rs_robust
```

组合结果：

| 组合 | 年化收益 | Sharpe | 最大回撤 | Calmar | 最终净值 |
|---|---:|---:|---:|---:|---:|
| 全品种等权 | 3.09% | 0.601 | 12.36% | 0.250 | 1.36 |
| 全品种波动率倒数 | 3.72% | 0.808 | 9.01% | 0.413 | 1.44 |

Ensemble 结论：

1. 多参数平均信号能降低单一参数过拟合，但收益被明显摊薄。
2. 缩窄参数族后，有色板块改善，但全品种组合提升不明显。
3. 当前 ensemble 版本可以作为探索结果，但暂时不如固定参数 Top2 清晰。

## 8. 固定参数样本外结果

样本外区间为：

```text
2024-06-07 ~ 2026-06-16
```

### 8.1 Top1 固定参数样本外

参数：

```text
short=20, long=120, dev=40, hurst=rs_fast
```

| 组合 | 区间 | 年化收益 | Sharpe | 最大回撤 | Calmar | 最终净值 |
|---|---|---:|---:|---:|---:|---:|
| 全品种等权 | 样本内 | 5.05% | 0.631 | 12.28% | 0.411 | 1.642 |
| 全品种等权 | 样本外 | -2.87% | -0.381 | 9.96% | -0.288 | 0.945 |
| 全品种波动率倒数 | 样本内 | 6.42% | 0.850 | 8.90% | 0.722 | 1.871 |
| 全品种波动率倒数 | 样本外 | -1.82% | -0.270 | 8.35% | -0.219 | 0.965 |

Top1 样本外表现转弱。样本外相对较好的品种包括 JM、CU、TA，拖累较明显的是 I、J、AL、RU、ZN。

### 8.2 Top2 固定参数样本外

参数：

```text
short=60, long=140, dev=60, hurst=rs_default
```

| 组合 | 区间 | 年化收益 | Sharpe | 最大回撤 | Calmar | 最终净值 |
|---|---|---:|---:|---:|---:|---:|
| 全品种等权 | 样本内 | 6.08% | 0.925 | 14.04% | 0.433 | 1.811 |
| 全品种等权 | 样本外 | -0.84% | -0.123 | 11.42% | -0.074 | 0.984 |
| 全品种波动率倒数 | 样本内 | 7.65% | 1.020 | 12.83% | 0.596 | 2.100 |
| 全品种波动率倒数 | 样本外 | -2.44% | -0.302 | 13.07% | -0.187 | 0.953 |

Top2 的样本内表现更强，但样本外仍然转弱。若只看样本外，Top2 的等权组合相对更稳，亏损小于 Top1。

