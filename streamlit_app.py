# streamlit_app.py
# Streamlit app with INLINE CUSP‑T (no external imports required)
# License: Research / evaluation only. Validate before production use.

import io
import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ============================================================
# CUSP‑T (INLINE LIBRARY) — morphology-preserving translation
# ============================================================
# Notes:
# - No Streamlit/UI references in this block.
# - Helper names are prefixed with _cuspt_ to avoid collisions.

def _cuspt_split_by_wrap(df: pd.DataFrame):
    V = df['V'].values
    cuts = [0]
    for i in range(1, len(V)):
        if V[i] < V[i-1] - 1e-9:
            cuts.append(i)
    cuts.append(len(V))
    sweeps=[]
    for s,e in zip(cuts[:-1], cuts[1:]):
        seg = df.iloc[s:e].copy().reset_index(drop=True)
        if 'G' in seg.columns: seg['G'] = float(seg['G'].median())
        if 'T' in seg.columns: seg['T'] = float(seg['T'].median())
        sweeps.append(seg)
    return sweeps

def _cuspt_detect_kink_knots(V: np.ndarray, I: np.ndarray, window: int=9, thresh: float=5.0):
    n = len(V)
    if n < window + 2: return []
    k = max(3, window | 1)
    pad = k//2
    def smooth(x):
        xx = np.pad(x, (pad,pad), mode='edge')
        ker = np.ones(k)/k
        return np.convolve(xx, ker, mode='valid')
    Vs = smooth(V); Is = smooth(I)
    d1 = np.gradient(Is, Vs)
    d2 = np.gradient(d1, Vs)
    ref = np.median(np.abs(d2) + 1e-12)
    peaks = np.where(np.abs(d2) > thresh*ref)[0]
    if peaks.size == 0: return []
    picks = [int(peaks[0])]
    for i in range(1, len(peaks)):
        if peaks[i] != peaks[i-1] + 1:
            picks.append(int(peaks[i]))
        if len(picks) == 2: break
    return [float(Vs[j]) for j in picks]

def _cuspt_fit_monotone_spline_fc(x: np.ndarray, y: np.ndarray):
    """Monotone, shape‑preserving cubic (Fritsch–Carlson)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 2: raise ValueError("need >=2 points")
    h = np.diff(x)
    delta = np.diff(y)/np.where(h==0, 1e-12, h)
    m = np.zeros_like(x)
    m[0]=delta[0]; m[-1]=delta[-1]
    m[1:-1] = (delta[:-1]+delta[1:])/2.0
    # FC limiter
    for i in range(x.size-1):
        if delta[i] == 0.0:
            m[i]=0.0; m[i+1]=0.0
        else:
            a = m[i]/delta[i]; b = m[i+1]/delta[i]; s = a*a + b*b
            if s > 9.0:
                t = 3.0/np.sqrt(s)
                m[i]   = t*a*delta[i]
                m[i+1] = t*b*delta[i]
    def eval(u):
        u = np.asarray(u, float)
        u_cl = np.clip(u, x[0], x[-1])
        idx = np.searchsorted(x, u_cl) - 1
        idx = np.clip(idx, 0, x.size-2)
        x0=x[idx]; x1=x[idx+1]; y0=y[idx]; y1=y[idx+1]
        m0=m[idx]; m1=m[idx+1]
        h=(x1-x0); t=np.where(h==0, 0.0, (u_cl-x0)/h)
        t2=t*t; t3=t2*t
        h00=2*t3-3*t2+1; h10=t3-2*t2+t
        h01=-2*t3+3*t2; h11=t3-t2
        return h00*y0 + h10*h*m0 + h01*y1 + h11*h*m1
    return eval

def _cuspt_fit_powerlaw_shunt(V: np.ndarray, I: np.ndarray, v_max=2.0):
    V=np.asarray(V); I=np.asarray(I)
    Vabs=np.abs(V)
    m=(Vabs>1e-4)&(Vabs<v_max)
    if m.sum()<10: return 1e-5, 1.5
    x=np.log(Vabs[m]); y=np.log(np.abs(I[m])+1e-15)
    A=np.vstack([np.ones_like(x), x]).T
    c, mexp = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(np.exp(c)), float(mexp)

def _cuspt_estimate_rs(V: np.ndarray, I: np.ndarray, top_frac=0.1):
    n=len(V); 
    if n<10: return 0.2
    idx=np.argsort(I); sel=idx[int((1-top_frac)*n):]
    dI=I[sel]-I[sel].mean(); dV=V[sel]-V[sel].mean()
    denom=(dI**2).sum()
    if denom<1e-12: return 0.2
    slope=(dI*dV).sum()/denom
    return float(max(0.0, slope))

def _cuspt_fit_sunsvoc(G: np.ndarray, T: np.ndarray, Voc: np.ndarray, beta_V=-0.08):
    G=np.asarray(G); T=np.asarray(T); Voc=np.asarray(Voc)
    if G.size<2:
        return {'A': float(Voc.mean() if Voc.size else 40.0), 'B': 0.0, 'Tref': 25.0, 'beta_V': float(beta_V)}
    Tref=float(np.median(T))
    x=np.log(np.maximum(G,1e-3)); y=Voc
    A=np.vstack([np.ones_like(x), x]).T
    a,b=np.linalg.lstsq(A,y,rcond=None)[0]
    return {'A': float(a), 'B': float(b), 'Tref': Tref, 'beta_V': float(beta_V)}

def _cuspt_predict_voc(model: dict, G: float, T: float):
    a=model.get('A',40.0); b=model.get('B',0.0); beta_V=model.get('beta_V',-0.08)
    return float(a + b*np.log(max(1e-3,G)) + beta_V*(T-25.0))

def cuspt_translate_inline(light_df: pd.DataFrame,
                           dark_df: pd.DataFrame,
                           sv_df: pd.DataFrame | None,
                           G2: float = 1000.0,
                           T2: float = 25.0,
                           alpha_I: float = 0.0005,
                           beta_V: float = -0.08,
                           rs_cap: float = 0.35,
                           a_bounds=(1e-6,5e-3),
                           m_bounds=(1.1,2.0)):
    """
    Morphology‑preserving translation with a non‑ohmic shunt and Rs‑free (Suns‑Voc) anchoring.
    Returns: {"translated": df(V,I), "neutralized": df(V,I), "diagnostics": {...}}
    """
    # 1) Anchor sweep
    sweeps=_cuspt_split_by_wrap(light_df)
    if not sweeps: raise ValueError("No Light I‑V sweep found.")
    anchor=sweeps[0]
    V1=anchor['V'].values; I1=anchor['I'].values
    G1=float(anchor['G'].median()); T1=float(anchor['T'].median())

    # 2) Isc, Voc
    top=max(1,int(0.02*len(I1)))
    Isc1=float(np.sort(I1)[-top:].mean())
    idx0=int(np.argmin(np.abs(I1))); Voc1=float(V1[idx0])

    # 3) Morphology M(V)=I/Isc
    M=I1/max(1e-9, Isc1)
    order=np.argsort(V1); V1s=V1[order]; Ms=M[order]
    shape=_cuspt_fit_monotone_spline_fc(V1s, Ms)

    # 4) Kink detection
    knots=_cuspt_detect_kink_knots(V1, I1)
    Vk1=float(knots[0]) if knots else 0.6*Voc1
    Vk1=float(np.clip(Vk1, 0.2*Voc1, 0.9*Voc1)) if Voc1>0 else Vk1

    # 5) Dark fits
    a_sh, m_sh=_cuspt_fit_powerlaw_shunt(dark_df['V'].values, dark_df['I'].values)
    a_sh=float(np.clip(a_sh, a_bounds[0], a_bounds[1]))
    m_sh=float(np.clip(m_sh, m_bounds[0], m_bounds[1]))
    Rs_est=min(rs_cap, _cuspt_estimate_rs(dark_df['V'].values, dark_df['I'].values))

    # 6) Suns‑Voc model + target anchors
    if sv_df is not None and len(sv_df)>=2:
        sv_model=_cuspt_fit_sunsvoc(sv_df['G'].values, sv_df['T'].values, sv_df['Voc'].values, beta_V=beta_V)
        Voc2=_cuspt_predict_voc(sv_model, G2, T2)
    else:
        Voc2=Voc1 + beta_V*((T2-25.0)-(T1-25.0)) + 0.65*np.log((G2/max(1e-3,G1))+1.0)
        sv_model={'A':Voc1,'B':0.65,'Tref':T1,'beta_V':beta_V}

    Isc2=Isc1*(G2/G1)*(1.0+alpha_I*(T2-25.0))/(1.0+alpha_I*(T1-25.0))

    # 7) Local‑stiffness warp (piecewise)
    r = Vk1/max(1e-9,Voc1)
    Vk2 = r*Voc2
    V2_grid = np.linspace(0.0, max(V1.max(), Voc2), len(V1))
    def W_inv(v2):
        if v2<=Vk2:
            t=v2/max(1e-9,Vk2); return t*Vk1
        t=(v2-Vk2)/max(1e-9,(Voc2-Vk2)); return Vk1 + t*max(1e-9,(Voc1-Vk1))
    V1_back = np.array([W_inv(v) for v in V2_grid])

    # 8) As‑is + neutralized
    M_interp = np.clip(shape(V1_back), 0.0, 1.05)
    I2_shape = Isc2*M_interp
    Ish2 = a_sh*np.maximum(0.0, V2_grid)**m_sh
    I2_as_is = np.maximum(0.0, I2_shape - Ish2 - 0.02*Rs_est*V2_grid)
    I2_neu   = np.maximum(0.0, Isc2*M_interp - 0.02*min(1e-3, Rs_est)*V2_grid)

    # 9) KPIs (basic)
    def kpi(V,I):
        P=V*I; j=int(np.argmax(P))
        return {"Pmax": float(P[j]), "Vmp": float(V[j]), "Imp": float(I[j]),
                "Voc": float(V[np.argmin(np.abs(I))]), "Isc": float(I[0])}

    return {
        "translated":  pd.DataFrame({"V": V2_grid, "I": I2_as_is}),
        "neutralized": pd.DataFrame({"V": V2_grid, "I": I2_neu}),
        "diagnostics": {
            "Isc1": Isc1, "Voc1": Voc1, "Isc2": Isc2, "Voc2": Voc2,
            "Vk1": Vk1, "Vk2": Vk2, "a_sh": a_sh, "m_sh": m_sh, "Rs_est": Rs_est,
            "sunsvoc_model": sv_model,
            "kpi_translated":  kpi(V2_grid, I2_as_is),
            "kpi_neutralized": kpi(V2_grid, I2_neu)
        }
    }

# =========================
# END INLINE CUSP‑T LIBRARY
# =========================


# ======================================
# STREAMLIT APP (UI + orchestration)
# ======================================

# Page setup
st.set_page_config(page_title="CUSP‑T: Morphology‑Preserving I‑V Translation", layout="wide")
st.title("CUSP‑T — Morphology‑Preserving, Degradation‑Aware I‑V Translation")
st.caption("Upload Light I‑V, Dark I‑V, and optional Suns‑Voc. Get 'as‑is' and 'neutralized' curves at target conditions with knee/kink preservation and non‑ohmic shunt handling.")

# Sidebar — inputs
st.sidebar.header("Inputs")
f_light = st.sidebar.file_uploader("Light I‑V CSV (columns: V,I,G,T)", type=["csv"])
f_dark  = st.sidebar.file_uploader("Dark I‑V CSV (columns: V,I)", type=["csv"])
f_suns  = st.sidebar.file_uploader("Suns‑Voc CSV (optional: G,T,Voc)", type=["csv"])

st.sidebar.subheader("Target conditions")
target_G = st.sidebar.number_input("Target irradiance G₂ [W/m²]", 200.0, 1400.0, 1000.0, 10.0)
target_T = st.sidebar.number_input("Target temperature T₂ [°C]", -20.0, 100.0, 25.0, 0.5)

with st.sidebar.expander("CUSP‑T parameters"):
    alpha_I = st.number_input("α_I (Isc temp coeff) [1/°C]", value=0.0005, format="%.6f")
    beta_V  = st.number_input("β_V for Voc temperature shift [V/°C]", value=-0.08, format="%.3f")
    rs_cap  = st.number_input("Rs cap (proxy upper bound) [Ω‑module‑proxy]", value=0.35, format="%.3f")
    a_min   = st.number_input("a_sh min", value=1e-6, format="%.6f")
    a_max   = st.number_input("a_sh max", value=5e-3, format="%.6f")
    m_min   = st.number_input("m_sh min", value=1.1, format="%.2f")
    m_max   = st.number_input("m_sh max", value=2.0, format="%.2f")

with st.sidebar.expander("Optional: IEC‑like surrogate overlay"):
    show_iec = st.checkbox("Compute IEC‑like surrogate overlay", value=False)
    alpha_I_iec = st.number_input("IEC‑like α_I [1/°C]", value=0.0005, format="%.6f")
    beta_V_iec  = st.number_input("IEC‑like β_V [V/°C]", value=-0.08, format="%.3f")
    lnG_fac     = st.number_input("IEC‑like ln(G) voltage factor", value=0.7, format="%.2f")

with st.sidebar.expander("Download CSV templates"):
    st.download_button("Light I‑V template",
        data="V,I,G,T\n0.0,8.5,1000,25\n...\n42.5,0.0,1000,25\n", file_name="template_light.csv", mime="text/csv")
    st.download_button("Dark I‑V template",
        data="V,I\n-1.0,0.005\n-0.5,0.002\n...\n1.0,0.050\n", file_name="template_dark.csv", mime="text/csv")
    st.download_button("Suns‑Voc template",
        data="G,T,Voc\n600,25,39.6\n800,25,40.2\n1000,25,40.8\n", file_name="template_sunsvoc.csv", mime="text/csv")

run = st.sidebar.button("Run CUSP‑T", type="primary")

# Utils
REQUIRED_LIGHT = {"V", "I", "G", "T"}
REQUIRED_DARK  = {"V", "I"}
REQUIRED_SV    = {"G", "T", "Voc"}

def _load_csv(upload, required_cols):
    df = pd.read_csv(upload)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df.dropna().copy()

def _plot_ivpv(anchor_df, tr_df, neu_df, iec_df=None, title="CUSP‑T results"):
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

def _kpi(df: pd.DataFrame):
    P = df["V"].values * df["I"].values
    j = int(np.argmax(P))
    Voc = float(df["V"].values[np.argmin(np.abs(df["I"].values))])
    Isc = float(df["I"].values[0])
    return {"Pmax": float(P[j]), "Vmp": float(df["V"].values[j]), "Imp": float(df["I"].values[j]),
            "Voc": Voc, "Isc": Isc}

def _iec_like_translate(anchor: pd.DataFrame, G2: float, T2: float,
                        alpha_I=0.0005, beta_V=-0.08, lnG_fac=0.7):
    V1 = anchor["V"].values; I1 = anchor["I"].values
    G1 = float(anchor["G"].median()); T1 = float(anchor["T"].median())
    I2 = I1*(G2/G1)*(1.0+alpha_I*(T2-25.0))/(1.0+alpha_I*(T1-25.0))
    dV_T = beta_V*((T2-25.0)-(T1-25.0))
    dV_G = lnG_fac*np.log(max(1e-6,G2)/max(1e-6,G1) + 1.0)
    V2 = V1 + dV_T + dV_G
    I2 = np.maximum(0.0, I2)
    return pd.DataFrame({"V": V2, "I": I2})

# Main
if run:
    if not (f_light and f_dark):
        st.error("Please upload both Light I‑V and Dark I‑V CSVs. Suns‑Voc is optional.")
        st.stop()

    try:
        df_light = _load_csv(f_light, REQUIRED_LIGHT)
        df_dark  = _load_csv(f_dark,  REQUIRED_DARK)
        df_suns  = _load_csv(f_suns,  REQUIRED_SV) if f_suns else None
    except Exception as e:
        st.error(f"CSV error: {e}")
        st.stop()

    # Build an anchor (first sweep) for plotting & IEC‑like overlay
    V = df_light["V"].values
    cuts=[0]
    for i in range(1,len(V)):
        if V[i] < V[i-1] - 1e-9: cuts.append(i)
    cuts.append(len(V))
    anchor_df = df_light.iloc[cuts[0]:cuts[1]].copy().reset_index(drop=True)

    # Run CUSP‑T inline
    try:
        res = cuspt_translate_inline(
            df_light, df_dark, df_suns,
            G2=target_G, T2=target_T,
            alpha_I=alpha_I, beta_V=beta_V,
            rs_cap=rs_cap, a_bounds=(a_min, a_max), m_bounds=(m_min, m_max)
        )
    except Exception as e:
        st.exception(e); st.stop()

    tr_df = res["translated"]; neu_df = res["neutralized"]; diag = res["diagnostics"]

    # Optional IEC‑like surrogate overlay
    iec_df = None
    if show_iec:
        iec_df = _iec_like_translate(anchor_df, target_G, target_T,
                                     alpha_I_iec, beta_V_iec, lnG_fac)

    st.success("CUSP‑T translation complete.")
    fig = _plot_ivpv(anchor_df, tr_df, neu_df, iec_df=iec_df,
                     title=f"CUSP‑T → (G={target_G:.0f} W/m², T={target_T:.1f} °C)")
    st.pyplot(fig)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.subheader("Measured (anchor)")
        km = _kpi(anchor_df)
        st.metric("Pmax [W]", f"{km['Pmax']:.2f}")
        st.caption(f"Voc: {km['Voc']:.2f} V | Isc: {km['Isc']:.2f} A")

    with c2:
        st.subheader("Translated (as‑is)")
        kt = _kpi(tr_df)
        st.metric("Pmax [W]", f"{kt['Pmax']:.2f}")
        st.caption(f"Vmp: {kt['Vmp']:.2f} V | Imp: {kt['Imp']:.2f} A")

    with c3:
        st.subheader("Neutralized")
        kn = _kpi(neu_df)
        st.metric("Pmax [W]", f"{kn['Pmax']:.2f}")
        st.caption(f"Vmp: {kn['Vmp']:.2f} V | Imp: {kn['Imp']:.2f} A")

    with c4:
        if iec_df is not None:
            ki = _kpi(iec_df)
            st.subheader("IEC‑like (surrogate)")
            st.metric("Pmax [W]", f"{ki['Pmax']:.2f}")
            st.caption(f"Vmp: {ki['Vmp']:.2f} V | Imp: {ki['Imp']:.2f} A")
        else:
            st.subheader("IEC‑like")
            st.caption("Not computed")

    st.divider()
    st.subheader("Diagnostics (CUSP‑T)")
    st.json(diag)

    # Downloads
    def _csv_button(label, df, fname):
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(label=label, data=csv, file_name=fname, mime="text/csv")

    st.subheader("Download corrected data")
    colA, colB, colC = st.columns(3)
    with colA: _csv_button("Translated (as‑is) CSV", tr_df, "cuspt_translated.csv")
    with colB: _csv_button("Neutralized CSV", neu_df, "cuspt_neutralized.csv")
    if iec_df is not None:
        with colC: _csv_button("IEC‑like (surrogate) CSV", iec_df, "iec_surrogate.csv")

    # Bundle all
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("anchor_measured.csv", anchor_df.to_csv(index=False))
        z.writestr("cuspt_translated.csv", tr_df.to_csv(index=False))
        z.writestr("cuspt_neutralized.csv", neu_df.to_csv(index=False))
        if iec_df is not None:
            z.writestr("iec_surrogate.csv", iec_df.to_csv(index=False))
        z.writestr("cuspt_diagnostics.json", json.dumps(diag, indent=2))
    st.download_button("Download all (ZIP)", data=buf.getvalue(),
                       file_name="cuspt_results_bundle.zip", mime="application/zip")

# Help / notes
with st.expander("CSV schemas, tips & rationale"):
    st.markdown("""
**Light I‑V:** `V, I, G, T`  
- Multiple sweeps can be concatenated; the app splits by voltage wrap and uses the *first* sweep as the anchor.

**Dark I‑V:** `V, I`  
- Include ±(1–2) V around 0 V for stable non‑ohmic shunt fit; include a forward high‑current region for a coarse \(R_s\).

**Suns‑Voc (optional):** `G, T, Voc`  
- Anchors \(V_{oc}(G)\) in an \(R_s\)-free way for a stable right‑hand boundary.

**Outputs:**  
- **Translated (as‑is):** keeps measured degradation (non‑ohmic shunt, relaxed bypass effect via morphology).  
- **Neutralized:** suppresses shunt/bypass/inactive‑area effects for “what‑if healthy” analysis at the same \((G_2,T_2)\).
""")
