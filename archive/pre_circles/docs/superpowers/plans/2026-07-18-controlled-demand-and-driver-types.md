# 受控需求 + 真实司机类型改造 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用"主道+匝道分开的受控 flow"替换 randomTrips 全网随机需求，重建 light/medium/dense 三档并把 HDV 司机类型真实化，使三档场景成为可信的 All-HDV 基线与 RL 训练/评估场景。

**Architecture:** 新增 Python 需求生成器（输出内嵌 `vTypeDistribution` 的 rou.xml，主道 flow + 匝道 flow 分开，流率 veh/h 直接可控）；SUMO 配置加碰撞记录、禁用 teleport 和动态改道；通过需求扫描找到"自由流 / 中度拥堵 / 拥堵但不锁死"的三档流率。RL 环境停用 TraCI 端 `HDVTypeAssigner`（类型由 route 文件分布采样，All-HDV 基线与 RL 评估天然一致）。

**Tech Stack:** SUMO 1.27.1（CLI）、Python 3（stdlib xml.etree）、pytest。环境：conda `rl_driving`，`SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo`。

## Global Constraints

- 场景工程目录：`D:\research\highway_merge_shepherd`（下称 SUMO 目录）；RL 工程目录：`D:\research\traffic_shepherd_rl`（下称 RL 目录），所有 Python 代码和测试放 RL 目录。
- 路网关键 edge（`config/constants.py:57-62` 已引用，勿改）：主道 `1191554636`（2 车道 1145m，31.29 m/s）→ 汇入区 `1487606179`（3 车道 75m）→ 下游 `1487606178`（2 车道 478m）；匝道 `11553048#1`（1 车道 379m，22.22 m/s）。
- 新需求文件名：`osm.demand.{light,medium,dense}.rou.xml`；旧 `osm.passenger.*.trips.xml` 保留但不再被 sumocfg 引用。
- 三档验收标准（All-HDV，3 seeds）：light 平均 timeLoss ≤5s、0 碰撞 0 teleport；medium timeLoss 10~30s、碰撞 ≤1；dense timeLoss 30~60s、0 teleport、碰撞 ≤2。dense 绝不允许过饱和锁死。
- 每个任务完成后 commit（执行前需得到用户对 git commit 的一次性确认）。
- 所有改动后 `pytest tests/` 必须全绿（57 个现有测试 + 新增）。

---

### Task 1: 受控需求生成器 `tools/build_demand.py`

**Files:**
- Create: `D:\research\traffic_shepherd_rl\tools\build_demand.py`
- Test: `D:\research\traffic_shepherd_rl\tests\test_build_demand.py`

**Interfaces:**
- Produces: `build_routes_xml(mainline_vph, ramp_vph, duration_s, mixture, seed) -> str`（rou.xml 文本）；`MIXTURES` dict；`write_demand(level, mainline_vph, ramp_vph, duration_s, mixture, out_dir) -> Path`。Task 4 的扫描脚本 import 这些函数。

- [ ] **Step 1: 写失败测试** `tests/test_build_demand.py`

```python
import xml.etree.ElementTree as ET

from tools.build_demand import MIXTURES, build_routes_xml


def _parse(xml_text):
    return ET.fromstring(xml_text)


def test_routes_contain_two_flows_and_two_routes():
    root = _parse(build_routes_xml(2400, 600, 900, "standard", 42))
    flows = root.findall("flow")
    routes = root.findall("route")
    assert len(flows) == 2
    assert len(routes) == 2
    main = next(f for f in flows if f.get("id") == "mainline")
    ramp = next(f for f in flows if f.get("id") == "ramp")
    assert main.get("vehsPerHour") == "2400"
    assert ramp.get("vehsPerHour") == "600"
    assert main.get("end") == "900"
    route_edges = {r.get("id"): r.get("edges") for r in routes}
    assert route_edges["mainline_route"] == "1191554636 1487606179 1487606178"
    assert route_edges["ramp_route"] == "11553048#1 1487606179 1487606178"


def test_vtype_distribution_probabilities_sum_to_one():
    root = _parse(build_routes_xml(2400, 600, 900, "standard", 42))
    dist = root.find("vTypeDistribution")
    assert dist is not None and dist.get("id") == "hdv_mix"
    vtypes = dist.findall("vType")
    assert {v.get("id") for v in vtypes} == {"aggressive_car", "normal_car", "cautious_car"}
    total = sum(float(v.get("probability")) for v in vtypes)
    assert abs(total - 1.0) < 1e-9


def test_all_mixtures_valid():
    for name in MIXTURES:
        root = _parse(build_routes_xml(1000, 200, 600, name, 1))
        vtypes = root.find("vTypeDistribution").findall("vType")
        assert abs(sum(float(v.get("probability")) for v in vtypes) - 1.0) < 1e-9


def test_aggressive_speedfactor_realistic():
    root = _parse(build_routes_xml(2400, 600, 900, "standard", 42))
    vtypes = {v.get("id"): v for v in root.find("vTypeDistribution").findall("vType")}
    # 31.29 m/s 限速下 aggressive 期望速度不超过 ~34 m/s（约 122 km/h）
    assert float(vtypes["aggressive_car"].get("speedFactor")) <= 1.10
    # 三种类型都必须有换道行为参数
    for v in vtypes.values():
        assert v.get("lcCooperative") is not None
        assert v.get("impatience") is not None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /d/research/traffic_shepherd_rl && python -m pytest tests/test_build_demand.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'tools.build_demand'`）

- [ ] **Step 3: 实现生成器** `tools/build_demand.py`

```python
"""Controlled mainline+ramp demand generator for the merge scenario.

Replaces randomTrips network-wide random demand with two explicit flows
(mainline + on-ramp) so that bottleneck inflow is directly controllable in
veh/h. Driver heterogeneity is declared inline as a SUMO vTypeDistribution,
so pure-SUMO baselines and TraCI-controlled runs share the exact same
driver mixture with zero code.
"""

import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path

MAINLINE_EDGE = "1191554636"
MERGE_EDGE = "1487606179"
DOWNSTREAM_EDGE = "1487606178"
RAMP_EDGE = "11553048#1"

# Realistic driver types (Krauss car-following + LC2013 lane changing).
# speedFactor is relative to the 31.29 m/s mainline limit.
DRIVER_TYPES = {
    "aggressive_car": {
        "tau": "0.9",
        "sigma": "0.6",
        "speedFactor": "1.08",
        "accel": "3.0",
        "decel": "4.5",
        "minGap": "1.5",
        "lcCooperative": "0.3",
        "lcAssertive": "0.6",
        "impatience": "0.7",
    },
    "normal_car": {
        "tau": "1.1",
        "sigma": "0.4",
        "speedFactor": "1.0",
        "accel": "2.6",
        "decel": "4.5",
        "minGap": "2.5",
        "lcCooperative": "0.6",
        "impatience": "0.3",
    },
    "cautious_car": {
        "tau": "1.5",
        "sigma": "0.2",
        "speedFactor": "0.92",
        "accel": "2.0",
        "decel": "3.0",
        "minGap": "3.5",
        "lcCooperative": "1.0",
        "impatience": "0.0",
    },
}

# Mixture presets: [aggressive, normal, cautious]
MIXTURES = {
    "standard": [0.10, 0.70, 0.20],
    "aggressive_heavy": [0.30, 0.60, 0.10],
    "cautious_heavy": [0.05, 0.75, 0.20],
}

_TYPE_ORDER = ["aggressive_car", "normal_car", "cautious_car"]


def build_routes_xml(mainline_vph, ramp_vph, duration_s, mixture, seed):
    """Return rou.xml text with a vTypeDistribution and two flows."""
    probs = MIXTURES[mixture]
    root = ET.Element("routes")
    dist = ET.SubElement(root, "vTypeDistribution", id="hdv_mix")
    for vtype_id, prob in zip(_TYPE_ORDER, probs):
        attrs = dict(DRIVER_TYPES[vtype_id])
        attrs["id"] = vtype_id
        attrs["probability"] = f"{prob:.2f}"
        ET.SubElement(dist, "vType", **attrs)
    ET.SubElement(
        root, "route", id="mainline_route", edges=f"{MAINLINE_EDGE} {MERGE_EDGE} {DOWNSTREAM_EDGE}"
    )
    ET.SubElement(
        root, "route", id="ramp_route", edges=f"{RAMP_EDGE} {MERGE_EDGE} {DOWNSTREAM_EDGE}"
    )
    for flow_id, route_id, vph in (
        ("mainline", "mainline_route", mainline_vph),
        ("ramp", "ramp_route", ramp_vph),
    ):
        ET.SubElement(
            root,
            "flow",
            id=flow_id,
            type="hdv_mix",
            route=route_id,
            begin="0",
            end=str(int(duration_s)),
            vehsPerHour=str(int(vph)),
            departLane="best",
            departSpeed="max",
        )
    rough = ET.tostring(root, encoding="unicode")
    return minidom.parseString(rough).toprettyxml(indent="    ")


def write_demand(level, mainline_vph, ramp_vph, duration_s, mixture, out_dir):
    """Write osm.demand.<level>.rou.xml into out_dir; return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"osm.demand.{level}.rou.xml"
    path.write_text(
        build_routes_xml(mainline_vph, ramp_vph, duration_s, mixture, seed=42), encoding="utf-8"
    )
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--level", required=True)
    p.add_argument("--mainline-vph", type=int, required=True)
    p.add_argument("--ramp-vph", type=int, required=True)
    p.add_argument("--duration", type=int, default=900)
    p.add_argument("--mixture", default="standard", choices=sorted(MIXTURES))
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    print(write_demand(a.level, a.mainline_vph, a.ramp_vph, a.duration, a.mixture, a.out_dir))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /d/research/traffic_shepherd_rl && python -m pytest tests/test_build_demand.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add tools/build_demand.py tests/test_build_demand.py
git commit -m "feat: controlled mainline+ramp demand generator with vTypeDistribution"
```

---

### Task 2: RL 环境停用 HDVTypeAssigner（类型由 route 分布接管）

**Files:**
- Modify: `D:\research\traffic_shepherd_rl\env\core.py:198-200`
- Modify: `D:\research\traffic_shepherd_rl\config\defaults.py:67-69`
- Test: `D:\research\traffic_shepherd_rl\tests\test_env_config.py`（新建）

**Interfaces:**
- Consumes: Task 1 的 rou.xml（env 启动 SUMO 时由 sumocfg 加载，vType 自动分布）。
- Produces: 新 config 键 `assign_hdv_types`（bool，默认 `False`）。为 `False` 时 `reset()` 不创建 `HDVTypeAssigner`；`step()` 里既有 `if self._hdv_type_assigner is not None` 守卫（`env/core.py:390`）无需改动。

- [ ] **Step 1: 写失败测试** `tests/test_env_config.py`

```python
from config.defaults import DEFAULT_CONFIG


def test_route_distribution_is_default():
    # 司机类型默认由 route 文件的 vTypeDistribution 提供，
    # 不再由 TraCI 端 HDVTypeAssigner 重分配（保证 All-HDV 与 RL 评估一致）。
    assert DEFAULT_CONFIG["assign_hdv_types"] is False
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /d/research/traffic_shepherd_rl && python -m pytest tests/test_env_config.py -v`
Expected: FAIL（`KeyError: 'assign_hdv_types'`）

- [ ] **Step 3: 实现**

`config/defaults.py` 第 67-69 行区域，在 `"hdv_type_proportions"` 之前插入：

```python
    # Driver types come from the route file's vTypeDistribution (single source
    # of truth, shared by All-HDV baseline and RL runs). Set True only for
    # legacy randomTrips route files without a distribution.
    "assign_hdv_types": False,
    "hdv_type_proportions": [0.1, 0.7, 0.2],      # legacy TraCI assignment
    "hdv_type_seed": 42,                          # legacy TraCI assignment
```

`env/core.py:198-200` 改为：

```python
        if self.config.get("assign_hdv_types", True):
            self._hdv_type_assigner = HDVTypeAssigner(
                proportions=self.config["hdv_type_proportions"], seed=hdv_seed
            )
        else:
            self._hdv_type_assigner = None
```

- [ ] **Step 4: 跑测试**

Run: `cd /d/research/traffic_shepherd_rl && python -m pytest tests/test_env_config.py tests/ -v`
Expected: 全绿。若有测试断言"reset 后 assigner 非 None"或依赖 TraCI 重分配，按"新默认行为"更新该测试的 config（显式传 `assign_hdv_types=True` 走 legacy 路径）。

- [ ] **Step 5: Commit**

```bash
git add env/core.py config/defaults.py tests/
git commit -m "feat: driver types from route vTypeDistribution by default"
```

---

### Task 3: 三档 sumocfg 修复（碰撞记录、禁 teleport、去动态改道）

**Files:**
- Modify: `D:\research\highway_merge_shepherd\osm.light.sumocfg`、`osm.medium.sumocfg`、`osm.dense.sumocfg`
- Modify: `D:\research\highway_merge_shepherd\hdv_types.add.xml`（参数与生成器对齐，留作参考/legacy）

**Interfaces:**
- Consumes: Task 1 生成的 `osm.demand.{level}.rou.xml`（Task 4 扫描定稿后由 Task 5 正式生成；本任务先用临时流率 light=1400/300 让配置可冒烟验证）。
- Produces: 三个 sumocfg 的统一结构（下游所有任务和 `evaluate_sumo_baseline.py` 直接受益）。

- [ ] **Step 1: 生成临时需求文件用于冒烟**

Run: `cd /d/research/traffic_shepherd_rl && python tools/build_demand.py --level light --mainline-vph 1400 --ramp-vph 300 --duration 900 --out-dir ../highway_merge_shepherd`
Expected: 输出 `../highway_merge_shepherd/osm.demand.light.rou.xml`

- [ ] **Step 2: 改写三个 sumocfg**（以 light 为例，medium/dense 把 `light` 全部换成对应档名）

```xml
<?xml version="1.0" encoding="UTF-8"?>
<sumoConfiguration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="osm.net.xml.gz"/>
        <route-files value="osm.demand.light.rou.xml"/>
        <additional-files value="output.light.add.xml"/>
    </input>
    <output>
        <tripinfo-output value="tripinfos.light.xml"/>
        <statistic-output value="stats.light.xml"/>
        <collision-output value="collisions.light.xml"/>
    </output>
    <processing>
        <ignore-route-errors value="true"/>
        <collision.action value="warn"/>
        <time-to-teleport value="-1"/>
    </processing>
    <report>
        <verbose value="true"/>
        <duration-log.statistics value="true"/>
        <no-step-log value="true"/>
    </report>
    <gui_only>
        <gui-settings-file value="osm.view.xml"/>
    </gui_only>
</sumoConfiguration>
```

要点：route-files 指向新 rou.xml；`hdv_types.add.xml` 从 additional-files 移除（类型已内嵌 rou.xml）；`<routing>` 动态改道块整段删除（单一路径下无用且会稀释瓶颈）；`collision.action=warn` 只记录不挪车；`time-to-teleport=-1` 禁用卡死 teleport（锁死会表现为 TT 暴涨而非车辆消失）。

- [ ] **Step 3: 同步 `hdv_types.add.xml` 参数**（与 Task 1 的 DRIVER_TYPES 完全一致，文件头注释注明"legacy，正式类型定义在 osm.demand.*.rou.xml 内嵌分布中"）

- [ ] **Step 4: 冒烟验证 light**

Run: `cd /d/research/highway_merge_shepherd && sumo -c osm.light.sumocfg`
Expected: 退出码 0；生成 `tripinfos.light.xml` / `stats.light.xml` / `collisions.light.xml`；`stats.light.xml` 中 `teleports total="0"`。

- [ ] **Step 5: Commit**

```bash
cd /d/research/highway_merge_shepherd && git add -A && git commit -m "scenario: controlled demand configs, collision logging, no teleport/rerouting"
```
（若该目录不是 git 仓库则跳过 commit，RL 目录测试通过后统一提交。）

---

### Task 4: 需求扫描 `tools/scan_demand.py` 并选定三档流率

**Files:**
- Create: `D:\research\traffic_shepherd_rl\tools\scan_demand.py`
- Create: `D:\research\traffic_shepherd_rl\runs\demand_scan\`（输出目录）
- Test: `D:\research\traffic_shepherd_rl\tests\test_scan_demand.py`

**Interfaces:**
- Consumes: Task 1 的 `build_routes_xml`、SUMO 目录 `osm.net.xml.gz`。
- Produces: `scan_cell(mainline_vph, ramp_vph, seed, duration_s) -> dict`（含 `mean_time_loss`、`mean_tt`、`collisions`、`teleports`、`arrived`）；`runs/demand_scan/scan_results.json` + `scan_table.md`。

- [ ] **Step 1: 写失败测试**（解析函数的单测，不跑 SUMO）

```python
from tools.scan_demand import parse_sumo_stats

STATS = """<?xml version="1.0"?>
<statistics>
    <performance vehicles="100" inserted="100" running="0" waiting="0"/>
    <vehicleTripStatistics routeLength="1500.0" speed="25.0" duration="60.0"
        waitingTime="5.0" timeLoss="12.5" departDelay="1.0"/>
    <teleports total="0" jam="0" yield="0" wrongLane="0"/>
    <safety collisions="2" emergencyStops="1" emergencyBraking="3"/>
</statistics>
"""


def test_parse_sumo_stats():
    m = parse_sumo_stats(STATS)
    assert m["collisions"] == 2
    assert m["teleports"] == 0
    assert m["arrived"] == 100
    assert abs(m["mean_time_loss"] - 12.5) < 1e-9
    assert abs(m["mean_tt"] - 60.0) < 1e-9
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /d/research/traffic_shepherd_rl && python -m pytest tests/test_scan_demand.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现扫描脚本**

```python
"""Scan mainline inflow rates to find light/medium/dense sweet spots."""

import argparse
import json
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from tools.build_demand import build_routes_xml

SUMO_DIR = Path(__file__).resolve().parent.parent.parent / "highway_merge_shepherd"
NET = SUMO_DIR / "osm.net.xml.gz"


def parse_sumo_stats(stats_xml_text):
    root = ET.fromstring(stats_xml_text)
    perf = root.find("performance")
    trip = root.find("vehicleTripStatistics")
    tele = root.find("teleports")
    safe = root.find("safety")
    return {
        "arrived": int(perf.get("vehicles")) if perf is not None else 0,
        "mean_tt": float(trip.get("duration")) if trip is not None else None,
        "mean_time_loss": float(trip.get("timeLoss")) if trip is not None else None,
        "teleports": int(tele.get("total")) if tele is not None else 0,
        "collisions": int(safe.get("collisions")) if safe is not None else 0,
    }


def scan_cell(mainline_vph, ramp_vph, seed, duration_s=900):
    """Run one All-HDV SUMO cell directly via CLI flags; return metrics dict."""
    with tempfile.TemporaryDirectory() as tmp:
        rou = Path(tmp) / "demand.rou.xml"
        rou.write_text(
            build_routes_xml(mainline_vph, ramp_vph, duration_s, "standard", seed), encoding="utf-8"
        )
        stats = Path(tmp) / "stats.xml"
        cmd = [
            "sumo",
            "-n",
            str(NET),
            "-r",
            str(rou),
            "--statistic-output",
            str(stats),
            "--collision.action",
            "warn",
            "--time-to-teleport",
            "-1",
            "--seed",
            str(seed),
            "--no-step-log",
            "true",
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(SUMO_DIR))
        m = parse_sumo_stats(stats.read_text(encoding="utf-8"))
        m.update(mainline_vph=mainline_vph, ramp_vph=ramp_vph, seed=seed)
        return m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mainline-rates", default="1400,1800,2200,2600,3000,3400")
    p.add_argument("--ramp-vph", type=int, default=600)
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--duration", type=int, default=900)
    p.add_argument("--out-dir", default="runs/demand_scan")
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    rates = [int(x) for x in a.mainline_rates.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    for rate in rates:
        for seed in seeds:
            m = scan_cell(rate, a.ramp_vph, seed, a.duration)
            rows.append(m)
            print(
                f"main={rate:5d} seed={seed}  TT={m['mean_tt']:.1f}  "
                f"timeLoss={m['mean_time_loss']:.1f}  coll={m['collisions']}  "
                f"tele={m['teleports']}  arrived={m['arrived']}",
                flush=True,
            )
    (out / "scan_results.json").write_text(json.dumps(rows, indent=1))
    lines = [
        "| mainline vph | seed | mean TT | timeLoss | collisions | teleports | arrived |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in rows:
        lines.append(
            f"| {m['mainline_vph']} | {m['seed']} | {m['mean_tt']:.1f} "
            f"| {m['mean_time_loss']:.1f} | {m['collisions']} "
            f"| {m['teleports']} | {m['arrived']} |"
        )
    (out / "scan_table.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
```

注意：`parse_sumo_stats` 的 `performance`/`vehicleTripStatistics` 标签属性名以实际 SUMO 1.27.1 输出为准，实现时先用一个 cell 的真实 `stats.xml` 校准属性名（SUMO 可能输出 `speed`/`duration` 等），单测样本同步修正。

- [ ] **Step 4: 跑测试 + 跑扫描**

```bash
cd /d/research/traffic_shepherd_rl && python -m pytest tests/test_scan_demand.py -v
python tools/scan_demand.py --mainline-rates 1400,1800,2200,2600,3000,3400 --ramp-vph 600
```
Expected: 测试通过；`runs/demand_scan/scan_table.md` 产出 18 行结果。

- [ ] **Step 5: 按验收标准选档并记录**

选档标准（3 seeds 均值）：light timeLoss ≤5s 且 0 碰撞；medium timeLoss 10~30s；dense timeLoss 30~60s 且 0 teleport 且碰撞 ≤2。若 3400 仍不达 dense 标准，加扫 3800/4200；若 dense 档出现过饱和（arrived 明显小于流量×时长、TT 爆炸），往低回退。把结论追加写入 `runs/demand_scan/scan_table.md` 末尾（选定档位 + 理由）。

- [ ] **Step 6: Commit**

```bash
git add tools/scan_demand.py tests/test_scan_demand.py runs/demand_scan/
git commit -m "feat: demand scan tooling + tier selection results"
```

---

### Task 5: 定稿三档 + 全量验证 + 文档更新

**Files:**
- Create: `D:\research\highway_merge_shepherd\osm.demand.{light,medium,dense}.rou.xml`（用 Task 4 选定流率正式生成）
- Modify: `D:\research\highway_merge_shepherd\density_presets.txt`
- Modify: `D:\research\traffic_shepherd_rl\docs\PROJECT_GUIDE.md`（场景描述段落）
- Test: 全量 `pytest tests/`

**Interfaces:**
- Consumes: Task 4 选定的三档流率。
- Produces: 可被 `train.py --density`、`evaluate_policy.py --density`、`tools/batch_eval.py` 直接使用的三档场景。

- [ ] **Step 1: 正式生成三档需求**（流率以 Task 4 结论为准，下为占位示例格式）

```bash
cd /d/research/traffic_shepherd_rl
python tools/build_demand.py --level light  --mainline-vph <L> --ramp-vph 300 --duration 900 --out-dir ../highway_merge_shepherd
python tools/build_demand.py --level medium --mainline-vph <M> --ramp-vph 500 --duration 900 --out-dir ../highway_merge_shepherd
python tools/build_demand.py --level dense  --mainline-vph <D> --ramp-vph 700 --duration 900 --out-dir ../highway_merge_shepherd
```

- [ ] **Step 2: 三档 All-HDV 验收**

Run: `cd /d/research/highway_merge_shepherd && sumo -c osm.light.sumocfg && sumo -c osm.medium.sumocfg && sumo -c osm.dense.sumocfg`
Expected（对照 Global Constraints 验收标准）：light timeLoss ≤5s / 0 碰撞；medium 10~30s；dense 30~60s / 0 teleport / 碰撞 ≤2。不达标则回 Task 4 调整流率。

- [ ] **Step 3: 更新 `density_presets.txt`**（新表格：档位、主道/匝道 veh/h、时长 900s、mixture、实测 All-HDV 指标）和 `docs/PROJECT_GUIDE.md` 场景段落（说明需求已从 randomTrips 改为受控 flow，司机类型来自 rou.xml 内嵌分布）。

- [ ] **Step 4: 全量测试**

Run: `cd /d/research/traffic_shepherd_rl && python -m pytest tests/ -v`
Expected: 全部通过（57 + 新增）。`test_env.py`（light）与 `test_multi_agent.py`（dense）会真实启动 SUMO 跑新场景，确认无超时/无异常。

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "scenario: finalize light/medium/dense controlled demand tiers"
```

---

### Task 6: 评估公平性（All-HDV 混合司机验证 + 多种子 + 指标补全 + no_op 修复）

**Files:**
- Modify: `D:\research\traffic_shepherd_rl\evaluate_sumo_baseline.py`（多种子 + 类型分布验证）
- Modify: `D:\research\traffic_shepherd_rl\evaluate_policy.py:53-64`（no_op 陷阱修复）
- Modify: `D:\research\traffic_shepherd_rl\evaluation.py`、`tools\batch_eval.py`、`tools\compare_to_hdv.py`（指标管线）
- Test: `D:\research\traffic_shepherd_rl\tests\`（新增针对性测试）

**需求：**
1. **All-HDV 多种子**：`evaluate_sumo_baseline.py` 支持 `--seeds 42,43,44`（透传 SUMO `--seed`），输出每 seed 指标 + mean±std；司机类型分布因 rou.xml 内嵌 vTypeDistribution 自动生效，从 tripinfo 输出统计三类型车辆数验证 ≈10/70/20 并写入报告。
2. **指标补全**：对比管线每条记录增加 throughput（arrived veh/h）、mean speed、mean timeLoss、hard braking（减速度 < −4.5 m/s² 事件数，从 FCD 流式解析或逐步 TraCI 统计）、collisions（SUMO 1.27 TraCI `simulation.getCollisions()` 或 collision-output 文件）、teleports。All-HDV 与 RL 评估同一套指标定义。
3. **no_op 陷阱修复**：`evaluate_policy.py:53-64` 对非学习 baseline 的默认 env config 是 `continuous_acceleration`，`NoOpPolicy` 动作 0 变成锁速控制。修复：no_op 在 shepherd 控制模式 + non_intervention 下运行（还权 SUMO）。验收：light 档 no_op 与 All-HDV 的旅行时间差 < 1%。
4. **"paper win" 判定更新**：`tools/compare_to_hdv.py` 的 win 规则改为：效率（TT 或 throughput）改善 且 安全（TTC+collisions+hard braking）不恶化 且 Δ 超过 seed std。
5. 只做 light 档冒烟验证，不重跑全网格（全网格重跑归入后续训练阶段）。

**约束：** 不改动 `tools/scan_demand.py`、`runs/demand_scan/`（Task 4 并行进行中）；git add 只加本任务文件。

---

### Task 7: 动作引擎温和化 + 护栏删除（方案一，用户确认）

**Files:**
- Modify: `D:\research\traffic_shepherd_rl\env\actions.py`（ShepherdActionEngine 重构）
- Modify: `D:\research\traffic_shepherd_rl\env\core.py:121-134`（引擎构造参数）
- Modify: `D:\research\traffic_shepherd_rl\config\defaults.py:52-66`（模式参数）
- Test: `D:\research\traffic_shepherd_rl\tests\`（更新动作引擎相关测试）

**背景：** 现有引擎的 gap_creation/conservative_following/release 是瞬时加减速脉冲（`mode_gap_accel=-2.0` 等，一步 setSpeed 到位），多 CAV 同时执行会形成同步刹车墙。为此打的两道补丁——4 步自动消退（`actions.py:171-185` 的 hold_state 计数）和 40% 降级（`actions.py:207-237` 的 `_maybe_downgrade`）——造成"选择≠执行≠奖励认定"三者不一致（诊断报告 R4）。本任务把动作温和化到"任意比例 CAV 同时执行也不会制造停车波"，然后删除两道护栏，使选择恒等于执行。奖励函数和观测一字不改。

**需求：**
1. **温和化**：所有非零模式的目标速度改用 `_rate_limit`（已存在于 `actions.py:64`）限速逼近，舒适界限 `mode_max_decel=1.0`、`mode_max_accel=1.0`、`mode_deadband=0.5`（新 config 键，写进 `defaults.py`）：
   - gap_creation：目标 `v_ego × 0.90`（沿用 `speed_factors`），tau +0.6 不变；
   - conservative_following：目标 `v_ego × 0.95`，tau +0.4 不变；
   - release：目标 `v_ego × 1.05`（封顶 max_speed），tau −0.3 不变；
   - flow_stabilization：EMA 调速器保留，`flow_stab_a_max_dec` 默认 2.0→1.0，`flow_stab_a_max_acc` 1.0 不变。
   - `_safe_speed_from_leader` 保留（那是物理安全，不是隐藏协调）。
2. **删除护栏**：移除 `_maybe_downgrade`、`gap_soft_cap_frac`、`mode_hold_steps` 自动消退和 `hold_state`（`ema_state` 保留给 flow_stab）。模式持续执行直到 policy 改选；`non_intervention` 仍负责复位 tau + setSpeed(-1)。`ShepherdActionEngine.__init__` 签名去掉 `mode_hold_steps`、`gap_soft_cap_frac`；`env/core.py` 构造处同步。
3. **清理 config**：`defaults.py` 删除/标注 legacy：`mode_hold_steps`、`gap_soft_cap_frac`、`mode_gap_accel`、`mode_conservative_accel`、`mode_release_accel`（脉冲参数随 `_accel_pulse` 一起退役，`_accel_pulse`/`_release`/`_fallback_factor` 被新的统一 `_gentle_mode` 实现替代）。
4. **文档同步**：`actions.py:1-27` 模块 docstring 的模式表改为新物理语义。
5. **测试**：新增/更新断言——(a) 任意模式下每步速度变化幅度 ≤ `mode_max_decel×dt`（减速方向）；(b) 连续选择同一模式超过 4 步仍然持续执行（无消退）；(c) 附近 CAV 全是 gap 模式时本车 gap_creation 不被降级；(d) non_intervention 复位 tau 并 setSpeed(-1)。既有断言旧护栏行为的测试按新语义更新。
6. **验证**：`conda run -n rl_driving python -m pytest tests/` 全绿；另跑一个 light 档 env 冒烟（可用 `tests/test_env.py` 同款 config 起 50 步）确认 SUMO 联调无异常。

**约束：** 不改 `env/rewards.py`、`env/features.py`（奖励与观测零改动是本任务的前提）；不碰 `tools/scan_demand.py`、`runs/demand_scan/`、`evaluate_*.py`、`evaluation.py`、`tools/batch_eval.py`、`tools/compare_to_hdv.py`（Task 4/6 并行中）；git add 只加本任务文件。

---

## 后续（不在本计划范围）

- `train.py` 训练 episode 上限 500 步（`train.py:622`）在 900s 需求窗下会截断拥堵过程——归入路线图阶段 1 的"训练协议修复"任务（改 `--max-episode-steps` ≈ 1200 + 多密度训练）。
- All-HDV 3-seed 化、指标补全（throughput/collision 进对比管线）归入路线图阶段 0 的其余条目。
- mixture 三预设（standard/aggressive_heavy/cautious_heavy）的扫描实验归入阶段 3。

---

### Task 8: ego 观测加 time-to-merge（用户确认，Z-Merge 依据）

**Files:**
- Modify: `env/features.py`（EGO_DIM 3→4，ego 向量追加 TTM）
- Modify: `models.py`（docstring ego(3)→ego(4)，输入维度随 EGO_DIM 自动更新）
- Modify: `docs/PROJECT_GUIDE.md`（观测维度 111→112 描述）
- Test: `tests/`（观测维度相关断言同步）

**需求：**
1. ego 向量 `[lane, speed, accel]` → `[lane, speed, accel, ttm]`；`EGO_DIM` 3→4；`raw_obs_dim()` 自动 111→112。
2. TTM 计算：复用现有 `distance_to_merge(cav_id)`（`features.py:320-340`），`ttm = dist / max(speed, 0.5)`，clip 到 [0, 60]，再除以 60 归一化（值域 [0,1]）；下游 edge 为 0；TraCI 异常时按既有风格返回零向量。Z-Merge 依据：time to merging point 属车载传感器信息（onboard sensors），无通信距离限制——zone features 的 500m V2X 限制与邻居 200m 感知半径均不变。
3. 网络结构除 ego 输入层 `nn.Linear(EGO_DIM, 32)` 自动 3→4 外零改动：ego embedding 输出仍 32，zone 7→64 不动，总 embedding 128 不动。
4. 旧 checkpoint 不兼容（观测维度变化），属预期。
5. 测试：TTM 计算的正确性（正常速度、低速 clip、下游 edge、异常路径）；全量 pytest 全绿。

**约束：** 不改 rewards.py、actions.py、zone features、邻居特征；不顺手重构。

---

### Task 9: 奖励结构修复（R_safe 有界化 + R_fair 基线参照化 + γ=0.995，用户确认）

**Files:**
- Modify: `env/rewards.py`（`_safety` 邻居聚合、`_fairness` 参照化）
- Modify: `config/defaults.py`（新增 per-density r_ref 表）
- Modify: `train.py`（γ 0.99→0.995；按 density 设置 fairness_r_ref）
- Modify: `evaluate_policy.py`（按 density 设置 fairness_r_ref）
- Test: `tests/`（奖励相关测试更新 + 新增）

**需求：**
1. **R_safe 有界化 + 去密度依赖**：邻居部分从"每辆急刹/TTC −1"（无下界，dense 下 −20~−30/步）改为 `−2.0 × (急刹或TTC违规的邻居比例)`，有界 [−2,0]；ego 部分（自身急刹 −1、TTC −1、碰撞 −5）不变。总 R_safe 由无下界变为有界 [−9, 0]。
2. **R_fair 从纯惩罚改为对 All-HDV 基线的有正负改善**：P90 子项改为 `clip((r_ref − p90)/r_ref, −1, +1)`（邻居 HDV 延误好于基线 → 正奖励）；方差子项改为 `−min(var/r_ref², 1.0)` 保持纯惩罚但有界；两项仍按 w_p90/w_var=0.5/0.5 组合，框架与权重表不变。
3. **r_ref 标定**：从 `runs/task5_acceptance/baseline_{light,medium,dense}.json` 计算各档 All-HDV 平均**逐步** timeLoss 率（mean_time_loss / mean duration），写入 `config/defaults.py` 的 `FAIRNESS_R_REF_BY_DENSITY = {"light": …, "medium": …, "dense": …}`（注释注明数据来源）；train.py / evaluate_policy.py 按 density 注入 env config `fairness_r_ref`，默认保持 0.5 向后兼容。
4. **γ 0.99 → 0.995**（train.py PPO config 一处）。
5. 测试：同一违规比例不同邻居数 → 邻居惩罚相同（密度无关）；R_safe 任意场景下有界；R_fair 在 p90<r_ref 时为正、p90>r_ref 时为负、且两侧有界；γ 断言。既有奖励测试按新语义更新。
6. 文档：rewards.py 模块 docstring 公式同步；docs/PROJECT_GUIDE.md 奖励段同步。

**约束：** 五项奖励框架、reward_weights 权重表、C_ctrl、R_eff、R_stab 语义全部不动；不改 env/actions.py、env/features.py、models.py；不顺手重构。

---

### Task 10: critic 全 CAV 输入 + 邻居编码器对齐（用户确认）

**Files:**
- Modify: `models.py`（FeatureEncoder 邻居分支）
- Modify: `env/mappo.py`（打包溢出告警）
- Modify: `config/defaults.py`（max_cavs_critic 提升）
- Test: `tests/`（编码器与打包测试）

**需求：**
1. **邻居编码器对齐设计**（models.py FeatureEncoder，输出维度不变）：4 个 Linear(1→32) 替换为单一共享 `nbr_embed = Sequential(Linear(4,32), ReLU)`，逐邻居独立编码（输入 rel_dist/rel_speed/rel_lane/turn_signal，trust 仍作池化权重不进编码）；池化从 trust 加权 sum 改为 trust 加权 mean（`Σ(e·w)/max(Σw,1)`）。ego(4→32)、zone(7→64)、总 embedding 128 全部不变；MAPPO critic 的 forward_components 自动跟随。
2. **critic 全 CAV**：实测三档 × p20 下最大并发活跃 CAV 数（写个小脚本跑 env 统计，数值写进报告），`max_cavs_critic` 设为大于实测最大值的 2 的幂（如 64/96）；`pack_mappo_observation` 保持字典序确定性打包，但 CAV 数超过 max_cavs 时打 stderr warning（fail-loud，不再静默截断）。
3. 测试：邻居编码器（单 Linear+ReLU 结构、padding 槽权重 0 不影响池化、3 邻居与 20 邻居同交通状态池化结果同尺度即 mean 语义）；打包溢出告警触发；全量 pytest 全绿。
4. 文档：models.py 模块 docstring、docs/PROJECT_GUIDE.md 网络段同步。

**约束：** raw 观测 112 维布局不变；encoder 输出 32/32/64→128 不变；不改 env/rewards.py、env/actions.py、env/features.py；不改 train.py 超参。
