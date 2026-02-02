# damp_t_app.py
# Streamlit app for DAmP-T: Degradation-Aware multi-modal Physically-constrained Translation
# Author: (Your name / org)
# License: For research and evaluation only. Validate before production use.

import io
import json
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------
# App config
# ---------------------------
st.set_page_config(page_title="DAmP-T: Degradation-Aware IV Correction", layout="wide")
st.title("DAmP-T — Degradation-Aware multi-modal Physically-constrained Translation")
st.caption("Upload Light I-V, Dark I-V, and optional Suns-Voc CSVs. Get corrected 'as-is' and 'neutralized' curves at target conditions.")

# ---------------------------
# Utility: CSV loaders
# ---------------------------
REQUIRED_LIGHT = {"V", "I", "G", "T"}
REQUIRED_DARK = {"V", "I"}
REQUIRED_SV = {"G", "T", "Voc"}

def load_csv(upload, required_cols):
    df = pd.read_csv(upload)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.dropna().copy()
    return df

def split_light_curves(df_light):
    """Split concatenated light I-V into individual sweeps by detecting V wrap."""
    V = df_light["V"].values
    idx = [0]
    for i in range(1, len(V)):
        if V[i] < V[i-1] - 1e-9:
            idx.append(i)
    idx.append(len(V))
    curves = []
    for s, e in zip(idx[:-1], idx[1:]):
        seg = df_light.iloc[s:e].copy()
        # Force scalar G, T per sweep (median)
        Gm = float(seg["G"].median())
        Tm = float(seg["T"].median())
        seg["G"], seg["T"] = Gm, Tm
        curves.append(seg.reset_index(drop=True))
    return curves

# ---------------------------
# Diagnostics: kink detection (bypass signature)
# ---------------------------
def detect_kinks(V, I, window=9, thresh=5.0):
    """Simple 2nd-derivative peak finder; returns 0/1/2 approximate kink count."""
    if len(V) < window + 2:
        return 0
    # Smooth
    k = max(3, window | 1)
    pad = k // 2
    def smooth(x):
        xx = np.pad(x, (pad, pad), mode="edge")
        ker = np.ones(k) / k
        return np.convolve(xx, ker, mode="valid")
    Vs = smooth(V)
    Is = smooth(I)
    d1 = np.gradient(Is, Vs)
    d2 = np.gradient(d1, Vs)
    ref = np.median(np.abs(d2) + 1e-9)
    pk = int(np.sum(np.abs(d2) > (thresh * ref)))
    if pk > 2:
        return 2
    if pk > 0:
        return 1
    return 0

# ---------------------------
# Dark I-V fits: power-law shunt and coarse Rs
# ---------------------------
def fit_shunt_powerlaw(V, I, v_max=2.0):
    """Fit I ~ a*|V|^m in low-voltage region (dark IV); return dict(a, m)."""
    V = np.asarray(V); I = np.asarray(I)
    Vabs = np.abs(V)
    mask = (Vabs > 1e-4) & (Vabs < v_max)
    if mask.sum() < 10:
        return {"a": 1e-5, "m": 1.5}
    x = np.log(Vabs[mask])
    y = np.log(np.abs(I[mask]) + 1e-15)
    A = np.vstack([np.ones_like(x), x]).T
    c, m = np.linalg.lstsq(A, y, rcond=None)[0]
    a = float(np.exp(c))
    return {"a": a, "m": float(m)}

def estimate_series_resistance(V, I, top_frac=0.1):
    """Estimate Rs from dark I-V at high current (linearized slope dV/dI)."""
    V = np.asarray(V); I = np.asarray(I)
    n = len(V)
    if n < 10:
        return 0.2
    idx = np.argsort(I)
    sel = idx[int((1 - top_frac) * n):]
    dI = I[sel] - I[sel].mean()
    dV = V[sel] - V[sel].mean()
    denom = (dI**2).sum()
    if denom < 1e-12:
        return 0.2
    slope = (dI * dV).sum() / denom
    return float(max(0.0, slope))

# ---------------------------
# Suns-Voc hooks (coarse)
# ---------------------------
kB = 1.380649e-23
q = 1.602176634e-19

def fit_suns_voc(G, T, Voc, Ns=60):
    """Fit an effective n_eff and J0_ref at Tref (median T)."""
    G = np.asarray(G); T = np.asarray(T); Voc = np.asarray(Voc)
    if len(G) < 2:
        return {"n_eff": 1.5, "J0_ref": 1e-10, "Tref": 25.0, "c_isc": 8.0}
    Tref = float(np.median(T))
    Vt = Ns * kB * (Tref + 273.15) / q
    x = np.log(np.maximum(G, 1e-3))
    y = Voc
    A = np.vstack([np.ones_like(x), x]).T
    c0, c1 = np.linalg.lstsq(A, y, rcond=None)[0]
    n_eff = float(np.clip(c1 / max(1e-9, Vt), 1.0, 2.2))
    c_isc = 8.0  # A per 1000 W/m^2 as a coarse module prior
    Gm = float(np.median(G))
    Vocm = float(np.median(Voc))
    Isc = c_isc * (Gm / 1000.0)
    J0 = max(1e-15, Isc / (np.exp(Vocm / (n_eff * Vt)) - 1.0))
    return {"n_eff": n_eff, "J0_ref": float(J0), "Tref": Tref, "c_isc": c_isc}

# ---------------------------
# Device model (sub-modules + non-ohmic shunt + relaxed bypass)
# ---------------------------
def Vth(TC, Ns=60):
    return Ns * kB * (TC + 273.15) / q

class SubmoduleParams:
    def __init__(self):
        self.Rs = 0.3
        self.I01 = 1e-9
        self.n1 = 1.4
        self.I02 = 1e-7
        self.n2 = 2.0
        self.a_sh = 1e-5   # non-ohmic shunt coefficient
        self.m_sh = 1.5    # shunt exponent
        self.phi = 1.0     # photocurrent scale
        self.eta_inactive = 0.0
        self.b_bypass = 0.0  # relaxed bypass prob [0..1]

class EnvCoeffs:
    def __init__(self):
        self.alpha_I = 0.0005   # Isc temp coeff (1/C)
        self.beta_Voc = -0.002  # placeholder (V/C) at module scale

class DegradationState:
    def __init__(self, theta=1.0):
        self.theta = theta

def photocurrent(G, TC, sp: SubmoduleParams, env: EnvCoeffs, degr: DegradationState):
    return sp.phi * (G / 1000.0) * (1.0 + env.alpha_I * (TC - 25.0)) * (1.0 - sp.eta_inactive) * degr.theta

def shunt_current(Vk, sp: SubmoduleParams):
    if Vk <= 0:
        return 0.0
    return sp.a_sh * (Vk ** sp.m_sh)

def diode_currents(Vk, I, TC, sp: SubmoduleParams, Ns=60):
    Vt = Vth(TC, Ns)
    arg1 = (Vk + I * sp.Rs) / (sp.n1 * Vt)
    arg2 = (Vk + I * sp.Rs) / (sp.n2 * Vt)
    arg1 = np.clip(arg1, -50, 50); arg2 = np.clip(arg2, -50, 50)
    return sp.I01 * (np.exp(arg1) - 1.0) + sp.I02 * (np.exp(arg2) - 1.0)

def submodule_residual(I, Vk, G, TC, sp: SubmoduleParams, env: EnvCoeffs, degr: DegradationState, Ns=60):
    Vk_eff = (1.0 - sp.b_bypass) * Vk  # relaxed bypass reduces effective voltage
    Il = photocurrent(G, TC, sp, env, degr)
    Id = diode_currents(Vk_eff, I, TC, sp, Ns)
    Ish = shunt_current(Vk_eff, sp)
    return I - (Il - Id - Ish)

def module_current(V, G, TC, subs, env: EnvCoeffs, degr: DegradationState, Ns=60):
    """Solve module current for a given terminal V with series submodules."""
    I = 0.0
    for _ in range(60):
        Vsum = 0.0
        for sp in subs:
            # bisection on Vk to make submodule residual ~ 0 for current I
            lo, hi = -5.0, max(5.0, V + 5.0)
            for __ in range(30):
                mid = 0.5 * (lo + hi)
                f = submodule_residual(I, mid, G, TC, sp, env, degr, Ns)
                if f > 0:
                    hi = mid
                else:
                    lo = mid
            Vk = 0.5 * (lo + hi)
            Vsum += Vk + I * sp.Rs
        errV = Vsum - V
        I -= 0.5 * errV / max(1e-6, sum(sp.Rs for sp in subs))
        if abs(errV) < 1e-4:
            break
    return max(0.0, I)

def translate_curve(V_grid, G2, T2, subs, env, degr):
    I_out = [module_current(v, G2, T2, subs, env, degr) for v in V_grid]
    return np.array(I_out)

# ---------------------------
# Pipeline (DAmP-T)
# ---------------------------
def init_submodules(kinks=0, M=3):
    subs = []
    for i in range(M):
        sp = SubmoduleParams()
        if kinks >= 1 and i == (M - 1):
            sp.b_bypass = 0.5  # relaxed prior
        subs.append(sp)
    return subs

def fit_composite(anchor_df, Rs_est, shunt_dict, subs):
    a_sh, m_sh = shunt_dict["a"], shunt_dict["m"]
    for sp in subs:
        sp.Rs = max(0.0, Rs_est / len(subs))
        sp.a_sh = max(1e-8, a_sh / len(subs))
        sp.m_sh = float(m_sh)
    env = EnvCoeffs()
    degr = DegradationState(theta=1.0)
    # Photocurrent scale from near short circuit region
    V = anchor_df["V"].values; I = anchor_df["I"].values
    Gm = float(anchor_df["G"].median()); Tm = float(anchor_df["T"].median())
    # take top 2% currents as Isc proxy
    top = max(1, int(0.02 * len(I)))
    Isc_proxy = float(np.sort(I)[-top:].mean())
    phi = (Isc_proxy / (Gm / 1000.0)) / len(subs)
    for sp in subs:
        sp.phi = float(np.clip(phi, 0.1, 20.0))
    return subs, env, degr

def damp_t_pipeline(df_light, df_dark, df_sv=None, target_G=1000.0, target_T=25.0):
    # 1) Split and choose an anchor sweep (first sweep)
    sweeps = split_light_curves(df_light)
    anchor = sweeps[0]
    V_anchor = anchor["V"].values; I_anchor = anchor["I"].values

    # 2) Kink detection
    kx = detect_kinks(V_anchor, I_anchor)
    subs = init_submodules(kx, M=3)

    # 3) Dark IV fits
    sh = fit_shunt_powerlaw(df_dark["V"].values, df_dark["I"].values)
    Rs = estimate_series_resistance(df_dark["V"].values, df_dark["I"].values)

    # 4) Suns-Voc (optional) – currently used as a soft check; extend if needed
    if df_sv is not None:
        sv_fit = fit_suns_voc(df_sv["G"].values, df_sv["T"].values, df_sv["Voc"].values)
    else:
        sv_fit = {"n_eff": 1.5, "J0_ref": 1e-10, "Tref": 25.0, "c_isc": 8.0}

    # 5) Fit composite (quick-init); replace with robust optimizer if desired
    subs, env, degr = fit_composite(anchor, Rs, sh, subs)

    # 6) Translate to target
    V_grid = np.linspace(float(anchor["V"].min()), float(anchor["V"].max()), 200)
    I_as_is = translate_curve(V_grid, target_G, target_T, subs, env, degr)

    # 7) Neutralized curve: remove bypass and shunt degradation
    subsN = []
    for sp in subs:
        s2 = SubmoduleParams()
        s2.__dict__.update(sp.__dict__)
        s2.b_bypass = 0.0
        s2.eta_inactive = 0.0
        s2.a_sh = min(s2.a_sh, 1e-8)
        subsN.append(s2)
    I_neutral = translate_curve(V_grid, target_G, target_T, subsN, env, degr)

    # 8) Package results
    out = {
        "anchor": anchor,
        "translated": pd.DataFrame({"V": V_grid, "I": I_as_is}),
        "neutralized": pd.DataFrame({"V": V_grid, "I": I_neutral}),
        "kinks_detected": int(kx),
        "Rs_estimate": float(Rs),
        "shunt_powerlaw": sh,
        "suns_voc_fit": sv_fit,
        "subs": [sp.__dict__ for sp in subs],
    }
    return out

# ---------------------------
# Plotting helpers
# ---------------------------
def plot_results(anchor_df, tr_df, neu_df, title="DAmP-T Results"):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    # IV
    ax[0].plot(anchor_df["V"], anchor_df["I"], "k.", label="Measured (anchor)")
    ax[0].plot(tr_df["V"], tr_df["I"], "C0-", label="Translated (as-is)")
    ax[0].plot(neu_df["V"], neu_df["I"], "C1--", label="Neutralized")
    ax[0].set_xlabel("Voltage [V]"); ax[0].set_ylabel("Current [A]"); ax[0].grid(True)
    ax[0].legend(); ax[0].set_title("I-V")

    # PV
    ax[1].plot(anchor_df["V"], anchor_df["V"] * anchor_df["I"], "k.", label="Measured")
    ax[1].plot(tr_df["V"], tr_df["V"] * tr_df["I"], "C0-", label="Translated")
    ax[1].plot(neu_df["V"], neu_df["V"] * neu_df["I"], "C1--", label="Neutralized")
    ax[1].set_xlabel("Voltage [V]"); ax[1].set_ylabel("Power [W]"); ax[1].grid(True)
    ax[1].legend(); ax[1].set_title("P-V")

    fig.suptitle(title); fig.tight_layout()
    return fig

def kpi(curve_df):
    P = curve_df["V"] * curve_df["I"]
    idx = int(np.argmax(P.values))
    return {
        "Pmax": float(P.iloc[idx]),
        "Vmp": float(curve_df["V"].iloc[idx]),
        "Imp": float(curve_df["I"].iloc[idx]),
    }

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("Inputs")
f_light = st.sidebar.file_uploader("Light I-V CSV (columns: V,I,G,T)", type=["csv"])
f_dark  = st.sidebar.file_uploader("Dark I-V CSV (columns: V,I)", type=["csv"])
f_suns  = st.sidebar.file_uploader("Suns-Voc CSV (optional: G,T,Voc)", type=["csv"])
target_G = st.sidebar.number_input("Target irradiance G2 [W/m^2]", 200.0, 1400.0, 1000.0, 10.0)
target_T = st.sidebar.number_input("Target temperature T2 [°C]", -20.0, 100.0, 25.0, 0.5)

run = st.sidebar.button("Run DAmP-T", type="primary")

# ---------------------------
# Main: Run pipeline
# ---------------------------
if run:
    if not (f_light and f_dark):
        st.error("Please upload both Light I-V and Dark I-V CSV files. Suns-Voc is optional.")
        st.stop()

    try:
        df_light = load_csv(f_light, REQUIRED_LIGHT)
        df_dark  = load_csv(f_dark, REQUIRED_DARK)
        df_suns  = load_csv(f_suns, REQUIRED_SV) if f_suns else None
    except Exception as e:
        st.error(f"CSV error: {e}")
        st.stop()

    try:
        res = damp_t_pipeline(df_light, df_dark, df_suns, target_G, target_T)
    except Exception as e:
        st.exception(e)
        st.stop()

    st.success("Translation complete.")
    fig = plot_results(res["anchor"], res["translated"], res["neutralized"], title="DAmP-T Translation to Target Conditions")
    st.pyplot(fig)

    # KPIs
    c1, c2, c3 = st.columns(3)
    k_meas = kpi(res["anchor"])
    k_tr   = kpi(res["translated"])
    k_neu  = kpi(res["neutralized"])

    with c1:
        st.subheader("Measured (anchor)")
        st.metric("Pmax [W]", f"{k_meas['Pmax']:.2f}")
        st.caption(f"Vmp: {k_meas['Vmp']:.2f} V  |  Imp: {k_meas['Imp']:.2f} A")

    with c2:
        st.subheader("Translated (as-is)")
        st.metric("Pmax [W]", f"{k_tr['Pmax']:.2f}")
        st.caption(f"Vmp: {k_tr['Vmp']:.2f} V  |  Imp: {k_tr['Imp']:.2f} A")

    with c3:
        st.subheader("Neutralized")
        st.metric("Pmax [W]", f"{k_neu['Pmax']:.2f}")
        st.caption(f"Vmp: {k_neu['Vmp']:.2f} V  |  Imp: {k_neu['Imp']:.2f} A")

    st.divider()
    st.subheader("Diagnostics & Parameters")
    st.json({
        "kinks_detected": res["kinks_detected"],
        "Rs_estimate": res["Rs_estimate"],
        "shunt_powerlaw": res["shunt_powerlaw"],
        "suns_voc_fit": res["suns_voc_fit"]
    })

    # Downloads
    def csv_button(label, df, fname):
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(label=label, data=csv, file_name=fname, mime="text/csv")

    st.subheader("Download corrected data")
    colA, colB = st.columns(2)
    with colA:
        csv_button("Download Translated (as-is) CSV", res["translated"], "translated_as_is.csv")
    with colB:
        csv_button("Download Neutralized CSV", res["neutralized"], "translated_neutralized.csv")

    # Bundle everything
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("anchor_measured.csv", res["anchor"].to_csv(index=False))
        z.writestr("translated_as_is.csv", res["translated"].to_csv(index=False))
        z.writestr("translated_neutralized.csv", res["neutralized"].to_csv(index=False))
        z.writestr("damp_t_params.json", json.dumps({
            "kinks_detected": res["kinks_detected"],
            "Rs_estimate": res["Rs_estimate"],
            "shunt_powerlaw": res["shunt_powerlaw"],
            "suns_voc_fit": res["suns_voc_fit"],
            "subs": res["subs"]
        }, indent=2))
    st.download_button("Download all (ZIP)", data=buf.getvalue(), file_name="damp_t_results_bundle.zip", mime="application/zip")

# ---------------------------
# Help / Notes
# ---------------------------
with st.expander("CSV schemas and tips"):
    st.markdown("""
**Light I-V CSV:** `V, I, G, T`  
- Voltage [V], Current [A], Irradiance G [W/m^2], Module temperature T [°C].  
- You can concatenate multiple sweeps; the app will split them by voltage wrap.

**Dark I-V CSV:** `V, I`  
- Forward/reverse dark sweep is fine; low-voltage region is used to fit the non-ohmic shunt power law.

**Suns-Voc CSV (optional):** `G, T, Voc`  
- If missing, the app uses a coarse default envelope.
  
**Outputs:**  
- **Translated (as-is):** keeps bypass, non-ohmic shunt, and degradation state.  
- **Neutralized:** bypass=off, inactive area=0, and shunt suppressed to a high-Rsh limit, useful for "what-if" comparisons.
""")
