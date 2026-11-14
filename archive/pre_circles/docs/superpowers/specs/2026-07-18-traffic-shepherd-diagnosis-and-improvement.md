# Traffic Shepherd 诊断报告与改进路线图

- 日期：2026-07-18
- 数据版本：`runs/eval/summary_all.json`（107 条记录，3 密度 × 3 渗透率 × 3 seeds × 4 baseline 全网格）、`runs/eval/diagnosis.md`
- 代码状态：git 有未提交改动（数值稳定性补丁 + `tools/compare_to_hdv.py`）
- 计划书：`Traffic-Shepherd-Short-Paper-en.pdf`

---

## 1. 执行摘要

当前所有学习型策略在有效意义上都跑不赢 All-HDV。**主因不是算法实现错误，而是三个协议级问题叠加**：

1. **场景错配**：训练只用 medium（接近自由流，理论改善空间 ~1.5s），而唯一有真实改善空间的 dense（All-HDV 有 41 起碰撞 + 41 次 teleport 卡死）从未用于训练；
2. **训练协议缺陷**：单点 medium/p10/seed42 训练、每 episode seed 固定、只跑了 1M 预算的 ~10%；
3. **评估协议不公平**：All-HDV 不做 HDV 司机类型混合、无种子方差、指标缺 throughput/collision/hard braking。

另有三个结构级问题会在协议修好后成为瓶颈：Shepherd 动作-奖励-执行不一致（dense 下 +38~+48s 灾难的直接原因）、观测缺 distance-to-merge、连续动作关闭 SUMO 安全层导致 TTC 暴涨。

改进路线分四个阶段：阶段 0 修评估（1-2 天）→ 阶段 1 重设计 dense 需求 + 修训练协议并重训（核心决策点）→ 阶段 2 修观测/动作/奖励结构（按需）→ 阶段 3 完整论文实验矩阵与消融。

---

## 2. 现状全景

### 2.1 需求档与 All-HDV 基线事实

需求档（`highway_merge_shepherd/density_presets.txt`）：

| 档位 | 车辆数/时长 | 约合流率 | All-HDV 平均 TT | TTC<1.5s 事件 | 碰撞 | teleport | 平均 timeLoss |
|---|---|---|---|---|---|---|---|
| light | 58 / 1200s | ~174 veh/h | 49.55s | 0 | 0 | 0 | ~1.0s |
| medium | 220 / 400s | ~1980 veh/h | 50.92s | 0 | **3** | 3 | ~9.0s |
| dense | 300 / 300s | ~3600 veh/h | 59.39s | 7 | **41** | **41** | — |

（碰撞/teleport 来自 `highway_merge_shepherd/stats.{light,medium,dense}.xml`，旅行时间来自 `runs/eval/all_hdv_*.json`）

关键读数：

- **light/medium 几乎没有效率改善空间**：medium 全车队平均 timeLoss 仅 ~9s，平摊到全车队可挤空间约 1.5s。在这里"跑赢"只能是噪声级数字。
- **dense 已过饱和**：3600 veh/h 超过该汇入区通行能力，All-HDV 产生 41 起碰撞、41 辆车卡死被 teleport。这既是灾难也是机会——**dense 有巨大的安全改善空间**（41 起碰撞），但效率空间受几何约束。
- **碰撞数目前完全不在 RL 评估指标里**（`summary_all.json` 只有 travel time + TTC 事件），最 dramatic 的 baseline 弱点根本没被量化。

### 2.2 策略 vs All-HDV（3 seeds 均值，ΔTT = 策略 − HDV，负为好）

完整数据见 `runs/eval/diagnosis.md`。要点：

| 策略 | light | medium | dense |
|---|---|---|---|
| rule_based | +0.2s（无效果） | +1.7s（更差） | **+8.6s**（更差，TTC 略降） |
| selfish | −0.55s（噪声级） | +0.6~2.7s，TTC +8~31 | p20 −4.7s 但 TTC +53.7 |
| flat_marl | −0.62s（噪声级） | p20 −1.5s 但 TTC +23.7 | p20 −6.4s 但 TTC +57.3 |
| shepherd | −0.08s（噪声级"win"） | +2.0~+4.1s，渗透率越高越差 | **+38~+48s，灾难性** |

模式总结：

- **连续动作策略（selfish/flat_marl）的效率-安全分裂**：dense/p20 能省 4.7~6.4s 行程时间，代价是 TTC 事件从 7 涨到 61~64。直接原因是 `rl_speed_mode = 6`（`config/defaults.py:84`，`config/constants.py:12-16`）关闭了 SUMO 的安全跟车速度检查。
- **Shepherd 的渗透率倒挂**：medium 下 p05/p10/p20 的 ΔTT = +1.97/+2.47/+4.07s，ramp TT 从 39.1s 恶化到 46.9s——CAV 越多，造间隙/保守跟车对主流的拖累越大。这与论文故事完全相反，是最需要解释和修复的现象。
- **Shepherd 在 dense 崩溃**（+38~+48s，seed 间 std 高达 41s）：间隙创建脉冲（`mode_gap_accel=-2.0`）+ tau 增大在过饱和流中触发主线停车波并锁死匝道。这说明当前 Shepherd 动作引擎（`env/actions.py` 的 `ShepherdActionEngine`）是为自由流设计的，在拥堵流中是有害的。
- `diagnosis.md` 里 shepherd light/p20 的 "PAPER_WIN"（−0.08s）是噪声级数字（seed std ±0.56s），其"推荐在 light 重训"的结论是误导——light 没有可赢的空间。

### 2.3 训练状态

| baseline | iterations | policy_reward_mean | 状态 |
|---|---|---|---|
| selfish | 16 | −7.96 → +2.83 | 基本收敛 |
| flat_marl | 20 | −8.48 → +15.53 | **仍在上升，未收敛** |
| shepherd | 16 | −71.3 → +15.46 | 接近平台 |

三个 run 各约 102k 步 = 默认 1M 预算的 ~10%（`train.py` 默认 100 万步），约 220 episodes。训练吞吐 38~89 步/s（CPU，12 个 env runner 各起一个 SUMO 实例）。

---

## 3. 根因诊断（按优先级排序）

### R1【场景】训练场景没有改善空间，且与评估分布错配

- 训练固定 medium + p10 + seed42（`train_all.py` 未传 `--density`/`--cav-penetration`），评估却是 3 密度 × 3 渗透率 × 3 seeds。dense 全是 OOD 泛化。
- medium 接近自由流，"牧羊人"动作（减速造间隙）在自由流里只有成本没有收益——这解释了 medium 下 shepherd 渗透率倒挂。
- 证据：`env/core.py:193-200` — `reset(seed=None)` 时 CAV 选择（seed=42）和 HDV 类型分配（seed=42）**每个 episode 完全相同**，训练多样性只剩 SUMO 跟车噪声 sigma。

### R2【训练】训练量不足

- 只跑了 ~102k/1M 步（10%），flat_marl 奖励曲线仍单调上升。以 38~89 步/s 计，1M 步约 3~7 小时/run，跑满完全可行。

### R3【评估】对比协议不公平、指标不完整

- `evaluate_sumo_baseline.py`：All-HDV 是纯 SUMO 子进程，**不做 HDV 类型重分配**（全是 default vType），而 RL 评估用 10/70/20 的 aggressive/normal/cautious 混合（`vehicles.py:51`，`hdv_types.add.xml`）——交通组成不同。
- All-HDV 单次确定性运行无种子方差，RL 有 3 seeds。
- 指标只有 travel time（mean/P90，all/mainline/ramp）+ TTC<1.5s 事件数。计划书 §V 要求的 throughput、mean speed、delay、hard braking、collision、公平性、控制成本均未进对比。
- 陷阱：`evaluate_policy.py:53-64` 给非学习 baseline 的默认 env config 是 `continuous_acceleration`，`NoOpPolicy` 的动作 0 会变成"每步 setSpeed(当前速度)"的锁速控制而非还权 SUMO。当前网格没跑 no_op 所以没污染结果，补跑前必须修。

### R4【结构】Shepherd 动作-奖励-执行三者不一致

- `env/actions.py:178-183`：非零模式 4 步后自动消退为 non_intervention——policy 以为自己在持续干预，实际不是。
- `env/actions.py:207-237`：附近 CAV 造间隙比例 >40% 时强制降级为 flow_stabilization——物理执行的动作不是 policy 选的动作。
- `env/core.py:360-365`：control cost 按"**选择的** mode"扣分，不是"执行的 mode"。
- 后果：policy 学到的是对被篡改动作的响应，信用分配系统性失真。
- 动作引擎参数为自由流标定（`mode_gap_accel=-2.0`、`tau_deltas.gap_creation=+0.6`），在拥堵流中直接制造停车波——dense 崩溃的主因。

### R5【结构】观测缺关键状态

- ego 观测只有 `[lane_index, speed, accel]`（`env/features.py:447`），**CAV 不知道自己离汇入口的距离**——对汇入控制任务这是核心状态。
- zone features 在 CAV 距汇入口 >500m 时整段 pad 成常数（`features.py:353-366`），上游 CAV 对瓶颈完全"盲"。
- MAPPO critic 的 32-CAV 截断按**车辆 ID 字符串字典序**取前 32 个（`env/mappo.py:52`），dense/p20 有 ~60 个 CAV 时 critic 看到的是任意子集而非空间邻近子集。

### R6【安全】连续动作关闭 SUMO 安全层

- speedMode=6 下 CAV 不遵守安全跟车速度和路权（`config/constants.py:12-16`），dense/p20 下 TTC 暴涨到 64 起是直接后果。
- 评估无碰撞统计，安全风险被 TTC 单一指标掩盖。

### R7【奖励】奖励结构有局部最优陷阱和尺度问题

- 效率项 `R_eff = v_ego/35` 每步白拿（mode 0 也拿），"不干预"已接近局部最优——shepherd 学到的保守策略部分是奖励结构喂出来的。
- safety 项无下界（200m 内每辆邻居急刹/TTC 都 −1），训练初期 per-step 奖励 −71，优势方差巨大。
- fairness 的 `r_ref=0.5` 是拍脑袋常数（`rewards.py:216-250`）。
- γ=0.99 对 ~454 步 episode：CAV 干预对下游车辆的长尾收益（>100 步后）几乎被折扣掉，与"牧羊人"的全局目标存在结构性张力。

### R8【参照系】rule_based baseline 本身反效果

- 让行逻辑在所有密度都增加 TT（dense +8.6s）。作为"规则 CAV"参照的说服力有限——要么需求太低让行纯损失，要么实现过于保守。论文里这个 baseline 需要修好或替换，否则审稿人会质疑。

---

## 4. 改进路线图

### 阶段 0：评估协议修复（不改模型，预计 1~2 人天）

目标：让"赢/输"的判定本身可信。

| # | 改动 | 位置 | 验证标准 |
|---|---|---|---|
| 0.1 | All-HDV baseline 加 `--additional-files hdv_types.add.xml` 同款类型混合（10/70/20），跑 3 seeds | `evaluate_sumo_baseline.py` | All-HDV 各密度指标有 mean±std，交通组成与 RL 评估一致 |
| 0.2 | 指标补全：throughput（arrived veh/h）、mean speed、timeLoss/delay、hard braking（<−4.5 m/s² 次数）、collision 数、teleport 数 | `evaluate_policy.py` + `tools/batch_eval.py` + `tools/compare_to_hdv.py` | `summary_all.json` 每条记录含全部计划书 §V 指标 |
| 0.3 | 修 NoOpPolicy 锁速陷阱（非学习 baseline 用 `shepherd` 模式 + non_intervention，或直接 TraCI 不控车） | `evaluate_policy.py:53-64` | no_op 结果与纯 SUMO All-HDV 在容差内一致 |
| 0.4 | "paper win" 重定义：效率（TT 或 throughput）改善 **且** 安全（TTC+collision+hard braking）不恶化 **且** P90 delay 不恶化；显著性要求 Δ 超过 seed std | `tools/compare_to_hdv.py` | 判定规则写入诊断报告模板 |

注意：0.1 会改变 All-HDV 基线数字本身（类型混合后 medium/dense 的 TT 和碰撞会变），所有后续对比以新基线为准。

### 阶段 1：场景重设计 + 训练协议修复（核心，预计 3~5 人天 + 机时）

目标：回答唯一的问题——**在合适的战场上，现有算法能否达到 paper win？**

| # | 改动 | 位置 | 验证标准 |
|---|---|---|---|
| 1.1 | **需求扫描**：在 2000~3200 veh/h 之间以 ~200 veh/h 步长生成 6~8 个新需求档，跑 All-HDV（3 seeds），画出 TT / timeLoss / 碰撞 / teleport / 匝道排队随需求的曲线 | `highway_merge_shepherd/build*.bat` + `randomTrips.py` | 找到"甜区"：All-HDV 出现 stop-and-go（timeLoss 15~40s、有 braking wave）但 teleport=0、碰撞接近 0。预期在 medium(1980) 与 dense(3600) 之间 |
| 1.2 | 以甜区需求替换/新增 dense 档；保留 light/medium 作对照组 | `density_presets.txt` | 新 dense 满足上述甜区标准 |
| 1.3 | **训练随机化**：每 episode 随机化 CAV 选择 seed 和 HDV 类型 seed（从 env seed 派生，可复现） | `env/core.py:193-200` | 两个相同 env seed 的 run 结果一致；不同 episode 的 CAV 集合不同 |
| 1.4 | **训练分布对齐**：shepherd/flat_marl/selfish 都在新 dense（或多密度混合采样）+ p10 训练；训练 config 支持密度列表 | `train.py:618-663`、`train_all.py` | 训练曲线在 1M 步内收敛或明确平台 |
| 1.5 | **跑满预算**：3 个 baseline 各 1M 步（约 3~7 小时/run，可串行过夜或并行） | `train.py` | TensorBoard 曲线 + `runs/pipeline_status.txt` 无错误 |
| 1.6 | 用阶段 0 的新评估管线重跑全网格 | `tools/batch_eval.py` | 新 `diagnosis.md` |

**决策点 D1**（这是整个路线图的咽喉）：

- 若 shepherd 在新 dense 达到 paper win → 跳到阶段 3 做论文实验；
- 若 flat_marl/selfish 能赢但 shepherd 不能 → 进阶段 2 修结构（问题在动作抽象）；
- 若所有策略都不能赢 → 进阶段 2 前先做根因分析（用阶段 0 的丰富指标看瓶颈到底在哪：是观测、奖励还是场景仍无空间），必要时重新评估论文假设。

### 阶段 2：结构与奖励修复（按需，预计 5~8 人天）

按预期收益排序，逐项独立合入并单独评估（每项都是一次消融）：

| # | 改动 | 位置 | 预期收益 |
|---|---|---|---|
| 2.1 | 观测加 distance-to-merge / time-to-merge；zone features 由"500m 内有效否则 padding"改为"全程有效（按距离衰减或全程 V2X）" | `env/features.py:447,353-366` | 消除观测缺口，上游 CAV 能提前决策 |
| 2.2 | 动作-奖励-执行一致化：control cost 按执行 mode 计；4 步消退和 40% 降级规则要么写进观测（agent 知道剩余 hold 步数/附近 gap 比例）要么删除 | `env/actions.py:178-237`、`env/core.py:360-365` | 信用分配不再失真 |
| 2.3 | Shepherd 动作引擎按拥堵流重新标定：gap_creation 的 −2.0 m/s² 脉冲和 tau+0.6 分密度档（自由流/拥堵两套参数，或改为"目标速度差"式温和干预） | `env/actions.py`、`config/defaults.py:38-66` | 消除 dense 崩溃（+38s → 接近 0） |
| 2.4 | 连续动作恢复 SUMO 安全层：speedMode 从 6 改为保留安全跟车检查的模式，或对 RL 加速度做安全后处理（碰撞时间约束 clip） | `config/constants.py:12-16`、`env/actions.py` | TTC 暴涨消失（dense/p20 64 起 → 接近 baseline） |
| 2.5 | 奖励整形：效率项与是否干预解耦（改为系统级速度/通过量）；safety 项 clip 下界；fairness 的 r_ref 用 All-HDV 实测 timeLoss 标定 | `env/rewards.py:132-250`、`config/defaults.py:23-29` | "不干预即最优"陷阱消除，初期奖励方差收敛 |
| 2.6 | MAPPO critic 的 32-CAV 截断按 ego 空间距离排序而非 ID 字典序 | `env/mappo.py:52` | critic 输入与局部性假设一致 |

**决策点 D2**：每项改动单独在新 dense 上重训 + 评估，保留有效的、回滚无效的。全部完成后重跑全网格。

### 阶段 3：论文实验（预计 2~3 周机时为主）

| # | 实验 | 说明 |
|---|---|---|
| 3.1 | γ ∈ {0.99, 0.995, 0.999} 扫描 | 454 步 episode 下干预长尾收益的折扣问题（R7） |
| 3.2 | 课程学习：light → medium → dense 顺序微调 vs 直接 dense 训练 | 可选，若直接训练收敛慢 |
| 3.3 | 完整矩阵：3 密度 × 3 渗透率 × **≥5 seeds**，报告 mean±std + 显著性检验（Mann-Whitney U 或 t-test） | 论文主表 |
| 3.4 | 计划书 §VII 消融：去 fairness / 去 stability / 去 control-cost / 去结构化动作（shepherd vs flat_marl 同奖励） | 证明 human-centric 设计的作用 |
| 3.5 | lane-drop 第二场景泛化（README:152-156 自标的 missing 项） | 结构泛化证据 |
| 3.6 | 行为轨迹可视化：速度剖面、汇入间隙、braking wave、CAV 动作时序（计划书 §VIII 要求） | 证明策略真的像"牧羊人"而非 reward hacking |

### 工作量与机时汇总

- 阶段 0：1~2 人天，机时可忽略
- 阶段 1：3~5 人天 + 约 1~2 天机时（需求扫描 + 3 run × 1M 步）
- 阶段 2：5~8 人天 + 每次消融 3~7 小时机时
- 阶段 3：2~3 周机时为主（完整矩阵 + 消融 + 泛化）

---

## 5. 风险与对策

| 风险 | 可能性 | 对策 |
|---|---|---|
| 需求甜区不存在（1980 自由流 → 3600 过饱和之间没有渐变区） | 低-中 | 改几何（缩短加速车道）或改需求时程（脉冲式需求制造 transient 拥堵）；匝道汇入瓶颈的容量骤降是普遍现象，大概率存在甜区 |
| 阶段 1 后所有策略仍不能赢 | 中 | 这正是计划书 §VIII 预设的"弱可控性"风险场景；此时转向分析"渗透率/需求边界图"（何种条件下稀疏控制有效），本身即是论文贡献 |
| Shepherd 动作空间本质不如连续动作 | 中 | 阶段 2.3 重新标定后若仍输 flat_marl，论文故事改为"结构化动作以 X% 效率代价换取 Y% 安全/可解释性收益"，仍是有效贡献 |
| 改安全层后连续动作学不出效率优势 | 低 | 安全后处理（TTC clip）比 speedMode 全局开关更灵活，作为 2.4 备选 |
| SUMO 司机行为模型不真实 | 已声明 | 计划书 §VIII 已限定结论为 simulation-based mechanism evidence，不改 |

---

## 6. 论文故事定位

维持计划书第六章的贡献定位，主线收敛为：

> **稀疏 CAV 在接近通行能力的拥堵瓶颈处能同时改善效率与安全；在低需求下无效甚至有害。** 我们给出稀疏控制有效性的渗透率-需求边界，以及结构化 human-centric 动作相对连续控制的安全优势。

light/medium 的负面结果不是失败，而是边界分析的对照组。 shepherd 当前在 medium 的渗透率倒挂（CAV 越多越差）如果能在修复后反转为正斜率，本身就是论文里最有说服力的一张图。

---

## 附录 A：关键复现命令

```bash
# 训练（当前协议，阶段 1 会改）
python train.py --baseline shepherd          # 默认 medium/p10/1M 步
python train_all.py                          # 顺序跑 selfish/flat_marl/shepherd

# 评估
python evaluate_sumo_baseline.py             # All-HDV（纯 SUMO）
python tools/batch_eval.py                   # 全网格 → runs/eval/summary_all.json
python tools/compare_to_hdv.py               # → runs/eval/diagnosis.md

# 测试
pytest tests/                                # 57 个测试
```

环境：conda `rl_driving`，`SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo`，SUMO 1.27.1，Ray RLlib（旧 ModelV2 API stack，`train.py:314-317`），CPU-only。

## 附录 B：场景与代码关键位置速查

| 主题 | 位置 |
|---|---|
| 路网/需求生成 | `D:\research\highway_merge_shepherd\build*.bat`、`density_presets.txt`、`hdv_types.add.xml` |
| 主线/汇入/ramp edge | `config/constants.py:57-62`（`1191554636` → `1487606179`，ramp `11553048#1`） |
| 奖励权重 | `config/defaults.py:23-29`、`rewards.py:132-368` |
| Shepherd 动作引擎 | `env/actions.py`（`ShepherdActionEngine`）、参数 `config/defaults.py:38-66` |
| 观测 | `env/features.py`（111 维 = ego3 + zone7 + count1 + 20邻居×5） |
| MAPPO critic 打包 | `env/mappo.py:32-60` |
| 训练超参 | `train.py:618-663`（lr 3e-4, γ 0.99, clip 0.2, batch 6000, 12 runners） |

## 附录 C：废弃目录说明

`C:\Users\jpeng\bottleneck_shepherd` 是 7 月 8 日的单密度原型副本，代码中无任何引用，可忽略（建议归档删除以免混淆）。
