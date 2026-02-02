using_cuspt = (fit_mode == "CUSP‑T (Morphology)")
if using_cuspt:
    from cuspt import cuspt_translate  # ensure cuspt.py is in the same folder
    res = cuspt_translate(
        df_light, df_dark, df_suns,
        G2=target_G, T2=target_T,
        alpha_I=0.0005, beta_V=-0.08,
        rs_cap=0.35, a_bounds=(1e-6, 5e-3), m_bounds=(1.1, 2.0)
    )
    translated = res["translated"]; neutralized = res["neutralized"]
    objective_info = None  # Not applicable for CUSP‑T proto
    # choose to display these instead of base['translated']/['neutralized']
