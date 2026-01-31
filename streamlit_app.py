
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bifacial IV Data Generator", layout="wide")
st.title("📈 Bifacial PV Module – Synthetic IV / Dark‑IV / Suns‑Voc Generator")
st.caption("Create reproducible, parameterized datasets for algorithm testing and DAmP‑T demos.")

# --------------------------
# Synthetic models (lightweight)
# --------------------------

def synth_light_iv_bifacial(Gf, Gr, gamma=0.7, T=25.0,
                             Voc_ref=41.0, Isc_ref=9.5, alpha_I=0.0005, beta_V=-0.08,
                             diode_knee=1.35, noise=0.0, Vmax=42.0, npts=220,
                             shunt_a=0.0, shunt_m=1.2, Rs=0.2, kink=None,
                             iam=1.0, seed=123):
    rng = np.random.default_rng(seed)
    V = np.linspace(0, Vmax, npts)
    Geff = iam*Gf + gamma*Gr
    Geff = max(1.0, Geff)
    Voc = Voc_ref + beta_V*(T-25.0) + 0.02*np.log(max(1e-6, Geff)/1000.0 + 1.0)
    Voc = max(5.0, Voc)
    Isc = Isc_ref*(Geff/1000.0)*(1.0 + alpha_I*(T-25.0))
    knee = max(1.05, diode_knee)
    I = Isc*np.maximum(0.0, 1.0 - (V/max(1e-6, Voc))**knee)
    I = np.maximum(0.0, I - Rs*V*0.02)
    if kink:
        V0 = float(kink.get('V0', Voc*0.55))
        depth = float(kink.get('depth', 0.25))*Isc
        width = float(kink.get('width', 0.9))
        I = np.maximum(0.0, I - depth*np.exp(-0.5*((V - V0)/width)**2))
    if shunt_a>0:
        Ish = shunt_a*np.maximum(0.0, V)**shunt_m
        I = np.maximum(0.0, I - Ish)
    if noise>0:
        I = np.maximum(0.0, I + rng.normal(0.0, noise, size=I.shape))
    df = pd.DataFrame({
        'V':V, 'I':I, 'G':np.full_like(V, Geff), 'T':np.full_like(V, T),
        'G_front':np.full_like(V, Gf), 'G_rear':np.full_like(V, Gr), 'gamma':np.full_like(V, gamma), 'IAM':np.full_like(V, iam)
    })
    return df


def synth_dark_iv(Rs=0.25, I01=1e-8, n_eff=1.9, shunt_a=0.0, shunt_m=1.2, Vmax=42.0, npts=320):
    V = np.linspace(0, Vmax, npts)
    Id = I01*(np.exp(V/(n_eff*1.0)) - 1.0)
    Ish = shunt_a*np.maximum(0.0, V)**shunt_m
    I = Id + Ish
    return pd.DataFrame({'V':V, 'I':I})


def synth_suns_voc_bifacial(Gf_list, Gr_list, gamma=0.7, T=25.0, Voc_stc=41.2, beta_V=-0.08):
    rows = []
    for Gf,Gr in zip(Gf_list, Gr_list):
        Geff = Gf + gamma*Gr
        Voc = Voc_stc + beta_V*(T-25.0) + 0.7*np.log(max(1e-3, Geff)/1000.0 + 1.0)
        rows.append({'G_front':Gf, 'G_rear':Gr, 'gamma':gamma, 'G':Geff, 'T':T, 'Voc':Voc})
    return pd.DataFrame(rows)

# --------------------------
# UI – Sidebar
# --------------------------
st.sidebar.header("Generator Settings")
mode = st.sidebar.radio("Mode", ["Preset scenarios", "Custom"], index=0)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=10_000_000, value=123, step=1)

# Common numeric controls
npts = st.sidebar.slider("Points per I-V curve", min_value=120, max_value=600, value=220, step=10)
Vmax = st.sidebar.slider("Max sweep voltage [V]", min_value=30.0, max_value=50.0, value=42.0, step=0.5)
noise = st.sidebar.slider("Measurement noise [A] (std dev)", 0.0, 0.05, 0.01, 0.001)

# --------------------------
# Presets
# --------------------------
if mode == "Preset scenarios":
    preset = st.selectbox(
        "Choose a scenario",
        [
            "Open‑yard (moderate albedo, 30°C)",
            "Rooftop low‑albedo (high IAM loss, 45°C)",
            "Snow/high‑albedo (0°C)",
            "Rear‑shaded kink (35°C)"
        ]
    )
    if preset == "Open‑yard (moderate albedo, 30°C)":
        params = dict(Gf=900, Gr=140, gamma=0.7, T=30, iam=0.98, Rs=0.18, shunt_a=1.5e-6, shunt_m=1.2, kink=None)
        sv = dict(Gf_list=[1000,900,800], Gr_list=[120,140,160], gamma=0.7, T=30, Voc_stc=41.4)
    elif preset == "Rooftop low‑albedo (high IAM loss, 45°C)":
        params = dict(Gf=800, Gr=60, gamma=0.7, T=45, iam=0.90, Rs=0.22, shunt_a=8e-4, shunt_m=1.3, kink=None)
        sv = dict(Gf_list=[850,800,700], Gr_list=[50,60,70], gamma=0.7, T=45, Voc_stc=41.0)
    elif preset == "Snow/high‑albedo (0°C)":
        params = dict(Gf=700, Gr=400, gamma=0.72, T=0, iam=0.99, Rs=0.16, shunt_a=1.0e-6, shunt_m=1.2, kink=None)
        sv = dict(Gf_list=[750,700,650], Gr_list=[350,400,450], gamma=0.72, T=0, Voc_stc=41.9)
    else:  # Rear‑shaded kink
        params = dict(Gf=950, Gr=120, gamma=0.7, T=35, iam=0.97, Rs=0.20, shunt_a=7e-4, shunt_m=1.3, kink={'V0':21.5,'depth':0.32,'width':0.8})
        sv = dict(Gf_list=[1000,950,900], Gr_list=[100,120,140], gamma=0.7, T=35, Voc_stc=41.2)

    if st.button("Generate datasets", type="primary"):
        light = synth_light_iv_bifacial(**params, noise=noise, Vmax=Vmax, npts=npts, seed=seed)
        dark  = synth_dark_iv(Rs=params['Rs'], shunt_a=params['shunt_a'], shunt_m=params['shunt_m'])
        suns  = synth_suns_voc_bifacial(**sv)

        st.success("Datasets generated.")
        # Plots
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(light['V'], light['I'], 'C0-')
            ax.set_xlabel('Voltage [V]'); ax.set_ylabel('Current [A]'); ax.set_title('Light I‑V (bifacial)')
            ax.grid(True)
            st.pyplot(fig)
        with col2:
            fig2, ax2 = plt.subplots(figsize=(6,4))
            ax2.plot(light['V'], light['V']*light['I'], 'C1-')
            ax2.set_xlabel('Voltage [V]'); ax2.set_ylabel('Power [W]'); ax2.set_title('P‑V')
            ax2.grid(True)
            st.pyplot(fig2)

        # Downloads
        def df_download(name, df):
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(label=f"Download {name} CSV", data=csv, file_name=f"{name}.csv", mime='text/csv')
        df_download('light_iv_bifacial', light)
        df_download('dark_iv_bifacial', dark)
        df_download('sunsvoc_bifacial', suns)

        # Zip bundle
        buf = io.BytesIO()
        import zipfile
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('light_iv_bifacial.csv', light.to_csv(index=False))
            z.writestr('dark_iv_bifacial.csv', dark.to_csv(index=False))
            z.writestr('sunsvoc_bifacial.csv', suns.to_csv(index=False))
        st.download_button("Download ZIP bundle", data=buf.getvalue(), file_name="bifacial_datasets.zip", mime="application/zip")

# --------------------------
# Custom mode
# --------------------------
else:
    st.subheader("Custom dataset")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        Gf = st.number_input("Front irradiance Gf [W/m²]", 0.0, 1400.0, 900.0, 10.0)
        iam = st.slider("IAM (front)", 0.5, 1.0, 0.98, 0.01)
    with c2:
        Gr = st.number_input("Rear irradiance Gr [W/m²]", 0.0, 800.0, 140.0, 10.0)
        gamma = st.slider("Bifaciality γ", 0.5, 1.0, 0.70, 0.01)
    with c3:
        T = st.number_input("Module temperature [°C]", -20.0, 90.0, 30.0, 1.0)
        Rs = st.slider("Series resistance proxy", 0.0, 0.5, 0.18, 0.01)
    with c4:
        shunt_a = st.number_input("Shunt a (A/V^m)", 0.0, 0.02, 0.0008, 0.0001, format="%0.4f")
        shunt_m = st.slider("Shunt exponent m", 1.0, 2.2, 1.3, 0.05)

    st.markdown("**Bypass‑kink (optional)**")
    kc1, kc2, kc3, kc4 = st.columns(4)
    with kc1:
        use_kink = st.checkbox("Enable kink", value=False)
    with kc2:
        V0 = st.number_input("Kink V0 [V]", 10.0, 30.0, 21.5, 0.1)
    with kc3:
        depth = st.slider("Kink depth (fraction of Isc)", 0.0, 0.8, 0.30, 0.01)
    with kc4:
        width = st.slider("Kink width [V]", 0.2, 2.0, 0.8, 0.05)

    kink = {'V0':V0, 'depth':depth, 'width':width} if use_kink else None

    if st.button("Generate custom datasets", type="primary"):
        light = synth_light_iv_bifacial(Gf=Gf, Gr=Gr, gamma=gamma, T=T, iam=iam, Rs=Rs,
                                        shunt_a=shunt_a, shunt_m=shunt_m, kink=kink,
                                        noise=noise, Vmax=Vmax, npts=npts, seed=seed)
        dark  = synth_dark_iv(Rs=Rs, shunt_a=shunt_a, shunt_m=shunt_m)
        # Build a small Suns‑Voc sweep around current conditions
        suns  = synth_suns_voc_bifacial([Gf, 0.9*Gf, 0.8*Gf], [Gr, Gr, Gr], gamma=gamma, T=T, Voc_stc=41.2)

        st.success("Datasets generated.")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(light['V'], light['I'], 'C0-')
            ax.set_xlabel('Voltage [V]'); ax.set_ylabel('Current [A]'); ax.set_title('Light I‑V (bifacial)')
            ax.grid(True)
            st.pyplot(fig)
        with col2:
            fig2, ax2 = plt.subplots(figsize=(6,4))
            ax2.plot(light['V'], light['V']*light['I'], 'C1-')
            ax2.set_xlabel('Voltage [V]'); ax2.set_ylabel('Power [W]'); ax2.set_title('P‑V')
            ax2.grid(True)
            st.pyplot(fig2)

        def df_download(name, df):
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(label=f"Download {name} CSV", data=csv, file_name=f"{name}.csv", mime='text/csv')
        df_download('light_iv_bifacial_custom', light)
        df_download('dark_iv_bifacial_custom', dark)
        df_download('sunsvoc_bifacial_custom', suns)

        buf = io.BytesIO()
        import zipfile
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('light_iv_bifacial_custom.csv', light.to_csv(index=False))
            z.writestr('dark_iv_bifacial_custom.csv', dark.to_csv(index=False))
            z.writestr('sunsvoc_bifacial_custom.csv', suns.to_csv(index=False))
        st.download_button("Download ZIP bundle", data=buf.getvalue(), file_name="bifacial_custom_datasets.zip", mime="application/zip")
