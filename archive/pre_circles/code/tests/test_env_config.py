from config.defaults import DEFAULT_CONFIG


def test_route_distribution_is_default():
    # 司机类型默认由 route 文件的 vTypeDistribution 提供，
    # 不再由 TraCI 端 HDVTypeAssigner 重分配（保证 All-HDV 与 RL 评估一致）。
    assert DEFAULT_CONFIG["assign_hdv_types"] is False
