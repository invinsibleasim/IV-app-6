# cuspt_app.py
# Streamlit app for CUSP‑T: Curvature‑ & Uncertainty‑aware, Shunt‑ & Probabilistic‑bypass‑preserving Translation
# License: Research / evaluation only. Validate before production use.

import io
import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---- Local CUSP‑T module ----
# Make sure cuspt.py (from cuspt_python_prototype.zip) is in the same folder.
from cuspt import cuspt_translate

st.set_page_config(page_title="CUSP‑T: Morphology‑Preserving I‑V Translation", layout="wide")
st.title("CUSP‑T — Morphology‑Preserving, Degradation‑Aware I‑V Translation")
st.caption("Upload Light I‑V, Dark I‑V, and optional Suns‑Voc. Get 'as‑is' and 'neutralized' curves at target conditions with knee/kink preservation and non‑ohmic shunt handling.")

# ---------------------------
# Sidebar — inputs
# ---------------------------
st.sidebar.header("Inputs")
f_light = st.sidebar.file_uploader("Light I‑V CSV (columns: V,I,G,T)", type=["csv"])
f_dark  = st.sidebar.file_uploader("Dark I‑V CSV (columns: V,I)", type=["csv"])
f_suns  = st.sidebar.file_uploader("Suns‑Voc CSV (optional: G,T,Voc)", type=["csv"])

st.sidebar.subheader("Target conditions")
target_G = st.sidebar.number_input("Target irradiance G₂ [W/m²]", 200.0, 1400.0, 1000.0, 10.0)
target_T = st.sidebar.number_input("Target temperature T₂ [°C]", -20.0, 100.0, 25.0, 0.5)

with st.sidebar.expander("Advanced (CUSP‑T)"):
    alpha_I = st.number_input("α_I (Isc temp coeff) [1/°C]", value=0.0005, format="%.6f")
    beta_V  = st.number_input("β_V for Voc temperature shift [V/°C]", value=-0.08, format="%.3f")
    rs_cap  = st.number_input("Rs cap (proxy upper bound) [Ω‑module‑proxy]", value=0.35, format="%.3f")
    a_min   = st.number_input("a_sh min", value=1e-6, format="%.6f")
    a_max   = st.number_input("a_sh max", value=5e-3, format="%.6f")
    m_min   = st.number_input("m_sh min", value=1.1, format="%.2f")
    m_max   = st.number_input("m_sh max", value=2.0, format="%.2f")

# Optional quick IEC‑like baseline for a visual compare
with st.sidebar.expander("Optional: Quick IEC‑like baseline overlay"):
    show_iec = st.checkbox("Compute IEC‑like surrogate overlay", value=False)
    alpha_I_iec = st.number_input("IEC‑like α_I [1/°C]", value=0.0005, format="%.6f")
    beta_V_iec  = st.number_input("IEC‑like β_V [V/°C]", value=-0.08, format="%.3f")
    lnG_fac     = st.number_input("IEC‑like ln(G) voltage factor", value=0.7, format="%.2f")

run = st.sidebar.button("Run CUSP‑T", type="primary")

# ---------------------------
# Helpers
# ---------------------------
REQUIRED_LIGHT = {"V","I","G","T"}
REQUIRED_DARK  = {"V","I"}
REQUIRED_SV    = {"G","T","Voc"}

def load_csv(upload, required_cols):
    df = pd.read_csv(upload)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df.dropna().copy()

def plot_ivpv(anchor_df, tr_df, neu_df, iec_df=None, title="CUSP‑T results"):
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    # I–V
    ax[0].plot(anchor_df["V"], anchor_df["I"], "k.", label="Measured (anchor)")
    ax[0].plot(tr_df["V"], tr_df["I"], "C0-", label="Translated (as‑is)")
    ax[0].plot(neu_df["V"], neu_df["I"], "C1--", label="Neutralized")
    if iec_df is not None:
        ax[0].plot(iec_df["V"], iec_df["I"], "C3:", label="IEC‑like (surrogate)")
    ax[0].set_xlabel("Voltage [V]"); ax[0].set_ylabel("Current [A]"); ax[0].grid(True); ax[0].legend()
    ax[0].set_title("I–V")

    # P–V
    ax[1].plot(anchor_df["V"], anchor_df["V"]*anchor_df["I"], "k.", label="Measured")
    ax[1].plot(tr_df["V"], tr_df["V"]*tr_df["I"], "C0-", label="Translated")
    ax[1].plot(neu_df["V"], neu_df["V"]*neu_df["I"], "C1--", label="Neutralized")
    if iec_df is not None:
        ax[1].plot(iec_df["V"], iec_df["V"]*iec_df["I"], "C3:", label="IEC‑like")
    ax[1].set_xlabel("Voltage [V]"); ax[1].set_ylabel("Power [W]"); ax[1].grid(True); ax[1].legend()
    ax[1].set_title("P–V")

    fig.suptitle(title); fig.tight_layout()
    return fig

def kpi(df: pd.DataFrame):
    P = df["V"].values * df["I"].values
    j = int(np.argmax(P))
    Voc = float(df["V"].values[np.argmin(np.abs(df["I"].values))])
    Isc = float(df["I"].values[0])
    return {
        "Pmax": float(P[j]),
        "Vmp": float(df["V"].values[j]),
        "Imp": float(df["I"].values[j]),
        "Voc": Voc,
        "Isc": Isc
    }

def iec_like_translate(anchor: pd.DataFrame, G2: float, T2: float,
                       alpha_I=0.0005, beta_V=-0.08, lnG_fac=0.7):
    V1 = anchor["V"].values; I1 = anchor["I"].values
    G1 = float(anchor["G"].median()); T1 = float(anchor["T"].median())
    I2 = I1*(G2/G1)*(1.0+alpha_I*(T2-25.0))/(1.0+alpha_I*(T1-25.0))
    dV_T = beta_V*((T2-25.0)-(T1-25.0))
    dV_G = lnG_fac*np.log(max(1e-6,G2)/max(1e-6,G1) + 1.0)
    V2 = V1 + dV_T + dV_G
    I2 = np.maximum(0.0, I2)
    return pd.DataFrame({"V":V2, "I":I2})

# ---------------------------
# Main run
# ---------------------------
if run:
    if not (f_light and f_dark):
        st.error("Please upload both Light I‑V and Dark I‑V CSVs. Suns‑Voc is optional.")
        st.stop()

    try:
        df_light = load_csv(f_light, REQUIRED_LIGHT)
        df_dark  = load_csv(f_dark,  REQUIRED_DARK)
        df_suns  = load_csv(f_suns,  REQUIRED_SV) if f_suns else None
    except Exception as e:
        st.error(f"CSV error: {e}")
        st.stop()

    try:
        res = cuspt_translate(
            df_light, df_dark, df_suns,
            G2=target_G, T2=target_T,
            alpha_I=alpha_I, beta_V=beta_V,
            rs_cap=rs_cap, a_bounds=(a_min, a_max), m_bounds=(m_min, m_max)
        )
    except Exception as e:
        st.exception(e); st.stop()

    tr_df = res["translated"]; neu_df = res["neutralized"]; diag = res["diagnostics"]

    # Anchor sweep (first sweep) for overlay
    # Split by voltage wrap: identical logic to cuspt._split_by_voltage_wrap
    V = df_light["V"].values
    cuts=[0]
    for i in range(1,len(V)):
        if V[i] < V[i-1] - 1e-9: cuts.append(i)
    cuts.append(len(V))
    anchor = df_light.iloc[cuts[0]:cuts[1]].copy().reset_index(drop=True)

    # Optional IEC‑like surrogate
    iec_df = None
    if show_iec:
        iec_df = iec_like_translate(anchor, target_G, target_T, alpha_I_iec, beta_V_iec, lnG_fac)

    st.success("CUSP‑T translation complete.")
    fig = plot_ivpv(anchor, tr_df, neu_df, iec_df=iec_df,
                    title=f"CUSP‑T → (G={target_G:.0f} W/m², T={target_T:.1f} °C)")
    st.pyplot(fig)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.subheader("Measured (anchor)")
        km = kpi(anchor)
        st.metric("Pmax [W]", f"{km['Pmax']:.2f}")
        st.caption(f"Voc: {km['Voc']:.2f} V | Isc: {km['Isc']:.2f} A")
    with c2:
        st.subheader("Translated (as‑is)")
        kt = kpi(tr_df)
        st.metric("Pmax [W]", f"{kt['Pmax']:.2f}")
        st.caption(f"Vmp: {kt['Vmp']:.2f} V | Imp: {kt['Imp']:.2f} A")
    with c3:
        st.subheader("Neutralized")
        kn = kpi(neu_df)
        st.metric("Pmax [W]", f"{kn['Pmax']:.2f}")
        st.caption(f"Vmp: {kn['Vmp']:.2f} V | Imp: {kn['Imp']:.2f} A")
    with c4:
        if iec_df is not None:
            st.subheader("IEC‑like (surrogate)")
            ki = kpi(iec_df)
            st.metric("Pmax [W]", f"{ki['Pmax']:.2f}")
            st.caption(f"Vmp: {ki['Vmp']:.2f} V | Imp: {ki['Imp']:.2f} A")
        else:
            st.subheader("IEC‑like")
            st.caption("Not computed")

    st.divider()
    st.subheader("Diagnostics (CUSP‑T)")
    st.json(diag)

    # Downloads
    def csv_button(label, df, fname):
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(label=label, data=csv, file_name=fname, mime="text/csv")

    st.subheader("Download corrected data")
    colA, colB, colC = st.columns(3)
    with colA: csv_button("Translated (as‑is) CSV", tr_df, "cuspt_translated.csv")
    with colB: csv_button("Neutralized CSV", neu_df, "cuspt_neutralized.csv")
    if iec_df is not None:
        with colC: csv_button("IEC‑like (surrogate) CSV", iec_df, "iec_surrogate.csv")

    # Bundle
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("anchor_measured.csv", anchor.to_csv(index=False))
        z.writestr("cuspt_translated.csv", tr_df.to_csv(index=False))
        z.writestr("cuspt_neutralized.csv", neu_df.to_csv(index=False))
        if iec_df is not None:
            z.writestr("iec_surrogate.csv", iec_df.to_csv(index=False))
        z.writestr("cuspt_diagnostics.json", json.dumps(diag, indent=2))
    st.download_button("Download all (ZIP)", data=buf.getvalue(),
                       file_name="cuspt_results_bundle.zip", mime="application/zip")

# ---------------------------
# Help / Notes
# ---------------------------
with st.expander("CSV schemas, tips & rationale"):
    st.markdown("""
**Light I‑V:** `V, I, G, T`  
- Multiple sweeps can be concatenated; the app splits by voltage wrap and uses the *first* sweep as anchor.

**Dark I‑V:** `V, I`  
- Include ±(1–2) V around 0 V for stable non‑ohmic shunt fit; include a forward high‑current region for a coarse \(R_s\).

**Suns‑Voc (optional):** `G, T, Voc`  
- Anchors \(V_{oc}(G)\) in an \(R_s\)-free way for a stable right‑hand boundary (recommended).

**Outputs:**  
- **Translated (as‑is):** keeps measured degradation (non‑ohmic shunt, relaxed bypass effect via morphology).  
- **Neutralized:** suppresses shunt/bypass/inactive‑area effects for “what‑if healthy” analysis at the same \((G_2,T_2)\).
""")
