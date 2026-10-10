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


def test_build_routes_xml_documents_seed_as_unused():
    # rou.xml is deterministic (flows + vTypeDistribution); all randomness is
    # driven by SUMO's own --seed at runtime. The docstring must say so, so
    # callers do not expect per-seed demand variation from this function.
    doc = build_routes_xml.__doc__ or ""
    assert "seed" in doc
    assert "unused" in doc.lower()
