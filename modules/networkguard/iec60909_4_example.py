"""The IEC TR 60909-4 example network (380/110/30/10 kV: grids, transformers, three-winding transformer, power station units,
generators, motors and lines) and its published short-circuit results, used to verify the NetworkGuard short-circuit engine.

Network model taken from the pandapower test suite (tests/shortcircuit/test_iec60909_4.py):
Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics and Energy System Technology (IEE),
Kassel. All rights reserved. Used under the pandapower BSD 3-clause licence."""
import numpy as np

from pandapower.create import create_empty_network, create_bus, create_ext_grid, create_transformer_from_parameters, \
    create_gen, create_transformer3w_from_parameters, create_line_from_parameters, create_motor

BUSES = ['Q (380 kV)', 'B2 (110 kV)', 'B3 (110 kV)', 'B4 (110 kV)', 'B5 (110 kV)', 'B6 (10 kV)', 'B7 (10 kV)', 'B8 (30 kV)', 'G1 terminal', 'G2 terminal']
# results published for the example (three-phase maximum, minimum and two-phase short circuits)
IK3_MAX = [40.6447, 31.7831, 19.6730, 16.2277, 33.1894, 37.5629, 25.5895, 13.5778, 52.4438, 80.5720]
IP3_MAX = [100.5677, 80.6079, 45.8111, 36.8427, 83.4033, 98.1434, 51.6899, 36.9227]
SK3_MAX = [26751.51, 6055.49, 3748.20, 3091.78, 6323.43, 650.61, 443.22, 705.52, 1816.71, 1395.55]
IK3_MIN = [5.0257, 12.0645, 10.2108, 9.3820, 11.6761, 27.7655, 18.3930, 10.9024, 44.4310, 67.8216]
IK2_MAX = [35.1994, 27.5249, 17.0373, 14.0536, 28.7429, 32.5304, 22.1611, 11.7586, 45.4177, 69.7774]
IP2_MAX = [87.0941, 69.8085, 39.6736, 31.9067, 72.2294, 84.9946, 44.7648, 31.9760, 118.0221, 182.1389]


def iec_60909_4():
    net = create_empty_network(sn_mva=34)

    b1 = create_bus(net, vn_kv=380.)
    b2 = create_bus(net, vn_kv=110.)
    b3 = create_bus(net, vn_kv=110.)
    b4 = create_bus(net, vn_kv=110.)
    b5 = create_bus(net, vn_kv=110.)
    b6 = create_bus(net, vn_kv=10.)
    b7 = create_bus(net, vn_kv=10.)
    b8 = create_bus(net, vn_kv=30.)

    HG1 = create_bus(net, vn_kv=20)
    HG2 = create_bus(net, vn_kv=10)  # 10.5kV?
    T_T5 = create_bus(net, vn_kv=10)
    T_T6 = create_bus(net, vn_kv=10)
    H = create_bus(net, vn_kv=30.)

    create_ext_grid(net, b1, s_sc_max_mva=38 * 380 * np.sqrt(3), rx_max=0.1, x0x_max=3, r0x0_max=0.15)
    create_ext_grid(net, b5, s_sc_max_mva=16 * 110 * np.sqrt(3), rx_max=0.1, x0x_max=3.3, r0x0_max=0.2)

    # t1 = create_transformer_from_parameters(net, b4, HG1, sn_mva=150,
    #     pfe_kw=0, i0_percent=0,
    #     vn_hv_kv=115., vn_lv_kv=21, vk_percent=16, vkr_percent=0.5,
    #     pt_percent=12, oltc=True)
    t1 = create_transformer_from_parameters(net, b4, HG1, sn_mva=150, pfe_kw=0, i0_percent=0, vn_hv_kv=115.,
                                            vn_lv_kv=21, vk_percent=16, vkr_percent=0.5, pt_percent=12, oltc=True,
                                            vk0_percent=15.2, vkr0_percent=0.5, xn_ohm=22, vector_group="YNd",
                                            mag0_percent=10000, mag0_rx=0, si0_hv_partial=0.5, power_station_unit=True)
    create_gen(net, HG1, p_mw=0.85 * 150, vn_kv=21, xdss_pu=0.14, rdss_ohm=0.002, cos_phi=0.85, sn_mva=150,
               pg_percent=0, power_station_trafo=t1)

    t2 = create_transformer_from_parameters(net, b3, HG2, sn_mva=100, pfe_kw=0, i0_percent=0, vn_hv_kv=120.,
                                            vn_lv_kv=10.5, vk_percent=12, vkr_percent=0.5, oltc=False, vk0_percent=12,
                                            vkr0_percent=0.5, vector_group="Yd", mag0_percent=10000, mag0_rx=0,
                                            si0_hv_partial=0.5, power_station_unit=True)
    create_gen(net, HG2, p_mw=0.9 * 100, vn_kv=10.5, xdss_pu=0.16, rdss_ohm=0.005, cos_phi=0.9, sn_mva=100,
               pg_percent=7.5, slack=True, power_station_trafo=t2)

    # # Add gen 3
    # create_gen(net, b6, p_mw=0.9 * 100, vn_kv=10.5,
    #               xdss_pu=0.1, rdss_ohm=0.018, cos_phi=0.8, sn_mva=10, pg_percent=5)
    # Add gen 3
    create_gen(net, b6, p_mw=0, vn_kv=10.5, xdss_pu=0.1, rdss_ohm=0.018, cos_phi=0.8, sn_mva=10, pg_percent=0)

    create_transformer3w_from_parameters(net, hv_bus=b1, mv_bus=b2, lv_bus=H, vn_hv_kv=400, vn_mv_kv=120, vn_lv_kv=30,
                                         sn_hv_mva=350, sn_mv_mva=350, sn_lv_mva=50, pfe_kw=0, i0_percent=0,
                                         vk_hv_percent=21, vkr_hv_percent=.26, vk_mv_percent=7, vkr_mv_percent=.16,
                                         vk_lv_percent=10., vkr_lv_percent=.16, vk0_hv_percent=44.1,
                                         vkr0_hv_percent=0.26, vk0_mv_percent=6.299627, vkr0_mv_percent=0.03714286,
                                         vk0_lv_percent=6.299627, vkr0_lv_percent=0.03714286, vector_group="YNyd",
                                         tap_changer_type="Ratio", tap_max=10, tap_min=-10, tap_pos=0, tap_neutral=0,
                                         tap_side="hv", tap_step_percent=0.1)
    # vk0 = sqrt(vkr0^2 + vki0^2) = sqrt(vkr^2 + (2.1 * vki)^2) = sqrt(vkr^2 + (2.1)^2 * (vk^2 - vkr^2))
    create_transformer3w_from_parameters(net, hv_bus=b1, mv_bus=b2, lv_bus=b8, vn_hv_kv=400, vn_mv_kv=120, vn_lv_kv=30,
                                         sn_hv_mva=350, sn_mv_mva=350, sn_lv_mva=50, pfe_kw=0, i0_percent=0,
                                         vk_hv_percent=21, vkr_hv_percent=.26, vk_mv_percent=7, vkr_mv_percent=.16,
                                         vk_lv_percent=10., vkr_lv_percent=.16, vk0_hv_percent=44.1,
                                         vkr0_hv_percent=0.26, vk0_mv_percent=6.299627, vkr0_mv_percent=0.03714286,
                                         vk0_lv_percent=6.299627, vkr0_lv_percent=0.03714286, vector_group="Yynd",
                                         tap_changer_type="Ratio", tap_max=10, tap_min=-10, tap_pos=0, tap_neutral=0,
                                         tap_side="hv", tap_step_percent=0.1)

    create_transformer3w_from_parameters(net, hv_bus=b5, mv_bus=b6, lv_bus=T_T5, vn_hv_kv=115., vn_mv_kv=10.5,
                                         vn_lv_kv=10.5, sn_hv_mva=31.5, sn_mv_mva=31.5, sn_lv_mva=31.5, pfe_kw=0,
                                         i0_percent=0, vk_hv_percent=12, vkr_hv_percent=.5, vk_mv_percent=12,
                                         vkr_mv_percent=.5, vk_lv_percent=12, vkr_lv_percent=.5, vk0_hv_percent=12,
                                         vkr0_hv_percent=0.5, vk0_mv_percent=12, vkr0_mv_percent=0.5, vk0_lv_percent=12,
                                         vkr0_lv_percent=0.5, vector_group="Yyd",  tap_changer_type="Ratio", tap_max=10,
                                         tap_min=-10, tap_pos=0, tap_neutral=0, tap_side="hv", tap_step_percent=0.1)
    create_transformer3w_from_parameters(net, hv_bus=b5, mv_bus=b6, lv_bus=T_T6, vn_hv_kv=115., vn_mv_kv=10.5,
                                         vn_lv_kv=10.5, sn_hv_mva=31.5, sn_mv_mva=31.5, sn_lv_mva=31.5, pfe_kw=0,
                                         i0_percent=0, vk_hv_percent=12, vkr_hv_percent=.5, vk_mv_percent=12,
                                         vkr_mv_percent=.5, vk_lv_percent=12, vkr_lv_percent=.5, vk0_hv_percent=12,
                                         vkr0_hv_percent=0.5, vk0_mv_percent=12, vkr0_mv_percent=0.5, vk0_lv_percent=12,
                                         vkr0_lv_percent=0.5, vector_group="Yynd", tap_changer_type="Ratio", tap_max=10,
                                         tap_min=-10, tap_pos=0, tap_neutral=0, tap_side="hv", tap_step_percent=0.1)
                                         # reactor is 100 Ohm

    create_motor(net, b7, pn_mech_mw=5.0, cos_phi=0.88, cos_phi_n=0.88, efficiency_n_percent=97.5, vn_kv=10, rx=0.1,
                 lrc_pu=5)
    for _ in range(2):
        create_motor(net, b7, pn_mech_mw=2.0, cos_phi=0.89, cos_phi_n=0.89, efficiency_n_percent=96.8, vn_kv=10, rx=0.1,
                     lrc_pu=5.2)

    create_line_from_parameters(net, b2, b3, name="L1", c_nf_per_km=0, max_i_ka=0,  # FIXME: Optional for SC
                                length_km=20, r_ohm_per_km=0.12, x_ohm_per_km=0.39, r0_ohm_per_km=0.32,
                                x0_ohm_per_km=1.26, c0_nf_per_km=0, g0_us_per_km=0)
    create_line_from_parameters(net, b3, b4, name="L2", c_nf_per_km=0, max_i_ka=0, length_km=10, r_ohm_per_km=0.12,
                                x_ohm_per_km=0.39, r0_ohm_per_km=0.32, x0_ohm_per_km=1.26, c0_nf_per_km=0,
                                g0_us_per_km=0)
    create_line_from_parameters(net, b2, b5, name="L3a", c_nf_per_km=0, max_i_ka=0, length_km=5, r_ohm_per_km=0.12,
                                x_ohm_per_km=0.39, r0_ohm_per_km=0.52, x0_ohm_per_km=1.86, c0_nf_per_km=0,
                                g0_us_per_km=0)
    create_line_from_parameters(net, b2, b5, name="L3b", c_nf_per_km=0, max_i_ka=0, length_km=5, r_ohm_per_km=0.12,
                                x_ohm_per_km=0.39, r0_ohm_per_km=0.52, x0_ohm_per_km=1.86, c0_nf_per_km=0,
                                g0_us_per_km=0)
    create_line_from_parameters(net, b5, b3, name="L4", c_nf_per_km=0, max_i_ka=0, length_km=10, r_ohm_per_km=0.096,
                                x_ohm_per_km=0.388, r0_ohm_per_km=0.22, x0_ohm_per_km=1.1, c0_nf_per_km=0,
                                g0_us_per_km=0)
    create_line_from_parameters(net, b5, b4, name="L5", c_nf_per_km=0, max_i_ka=0, length_km=15, r_ohm_per_km=0.12,
                                x_ohm_per_km=0.386, r0_ohm_per_km=0.22, x0_ohm_per_km=1.1, c0_nf_per_km=0,
                                g0_us_per_km=0)
    create_line_from_parameters(net, b6, b7, name="L6", c_nf_per_km=0, max_i_ka=0, length_km=1, r_ohm_per_km=0.082,
                                x_ohm_per_km=0.086, r0_ohm_per_km=0.082, x0_ohm_per_km=0.086, c0_nf_per_km=0,
                                g0_us_per_km=0)
    # bus F for 1ph fault: 1, 2, 3, 4
    return net
