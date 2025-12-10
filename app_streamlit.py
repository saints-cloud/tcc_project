
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from core.g4_export import build_zip

from core.logging_utils import setup_logging
from core.materials import MATERIALS, get_material
from core.spectrum import w_target_spectrum_tasmip_like
from core.spekpy_adapter import spekpy_available, generate_spectrum_spekpy
from core.attenuation import interpolate_mu, effective_mu
from core.xcom_builtin import elemental_mu_over_rho

st.set_page_config(page_title="MicroCT – Modelo Híbrido", layout="wide")
logger = setup_logging()

# ---------- Z_eff utilities ----------
ELEMENT_Z = {"H":1,"C":6,"N":7,"O":8,"Na":11,"Mg":12,"P":15,"S":16,"Cl":17,"K":19,"F":9,"Ca":20}

def compute_zeff(fractions: dict, m_exp: float = 2.94) -> float:
    s = 0.0
    for el, w in fractions.items():
        if w <= 0: 
            continue
        Z = ELEMENT_Z.get(el)
        if Z is None: 
            continue
        s += float(w) * (Z ** m_exp)
    if s <= 0:
        return float('nan')
    return s ** (1.0 / m_exp)

def get_fractions_for_material(name: str):
    from core.xcom_builtin import COMPOSITIONS
    if name in st.session_state.get("CUSTOM_MATERIALS", {}):
        return st.session_state.CUSTOM_MATERIALS[name]["fractions"]
    return COMPOSITIONS.get(name, {})

# ---------- Sidebar ----------
st.sidebar.header("Material e Geometria")
if "CUSTOM_MATERIALS" not in st.session_state:
    st.session_state.CUSTOM_MATERIALS = {}

st.sidebar.caption("Adicione materiais CUSTOM_ com frações mássicas e densidade na seção apropriada (abaixo, na página).")
materials_all = list(MATERIALS.keys()) + list(st.session_state.CUSTOM_MATERIALS.keys())
material = st.sidebar.selectbox("Material", materials_all, index=0)
thickness_mm = st.sidebar.slider("Espessura (mm)", 0.1, 30.0, 10.0, 0.1)

st.sidebar.subheader("Base XCOM (opcional)")
df_xcom = None
up = st.sidebar.file_uploader("CSV XCOM (keV + colunas minúsculas para materiais)", type=["csv"])
if up is not None:
    try:
        df_xcom = pd.read_csv(up)
        st.sidebar.success("CSV carregado.")
    except Exception as e:
        st.sidebar.warning(f"Falha ao ler CSV: {e}")

st.sidebar.subheader("Fonte / Espectro")
kvp = st.sidebar.slider("Tensão do tubo (kVp)", 20, 120, 80, 1)
filt_Al = st.sidebar.slider("Filtração (mm Al)", 0.0, 5.0, 1.5, 0.1)
filt_Cu = st.sidebar.slider("Filtração (mm Cu)", 0.0, 1.0, 0.0, 0.01)
use_spek = False
spec_source = "TASMIP-like"
if spekpy_available():
    use_spek = st.sidebar.toggle("Usar SpekPy (se instalado)", value=False)

# ---------- Energies & Spectrum ----------
E = np.linspace(10.0, float(kvp), 300)
if use_spek:
    try:
        spec = generate_spectrum_spekpy(kvp, E, filt_mm_Al=filt_Al, filt_mm_Cu=filt_Cu)
        spec_source = "SpekPy"
    except Exception as e:
        try:
            import spekpy as sp
            ver = getattr(sp, "__version__", "sem versão")
        except Exception:
            ver = "não detectado"
        st.sidebar.warning(f"SpekPy {ver} falhou: {e}. Usando TASMIP-like.")
        spec = w_target_spectrum_tasmip_like(kvp, E, filt_mm_Al=filt_Al, filt_mm_Cu=filt_Cu)
else:
    spec = w_target_spectrum_tasmip_like(kvp, E, filt_mm_Al=filt_Al, filt_mm_Cu=filt_Cu)

st.markdown("# MicroCT – Modelo Híbrido (v8)")

# ---------- Custom materials (inline) ----------
st.markdown(
    """
    <h3 style="text-align:center; color:#33C3F0; font-weight:700; margin-top:1.2em;">
        Materiais Personalizados
    </h3>
    """,
    unsafe_allow_html=True,
)


with st.expander("Adicionar/Editar Material CUSTOM_", expanded=False):
    cust_name = st.text_input("Nome (será convertido para UPPER e prefixo CUSTOM_ recomendado)", value="CUSTOM_MATERIAL")
    dens = st.number_input("Densidade (g/cm³)", value=1.00, step=0.01, min_value=0.1)
    st.write("Frações mássicas (somar ~1.0):")
    cols = st.columns(4)
    frac = {}
    for i, el in enumerate(["H","C","N","O","Na","Mg","P","S","Cl","K","F","Ca"]):
        with cols[i%4]:
            frac[el] = st.number_input(el, value=0.0, step=0.001, min_value=0.0, key=f"cust_{el}")
    st.caption(f"Soma = {sum(frac.values()):.3f}")
    if st.button("Salvar CUSTOM_"):
        nm = cust_name.strip().upper()
        if nm:
            st.session_state.CUSTOM_MATERIALS[nm] = {"density": float(dens), "fractions": {k:v for k,v in frac.items() if v>0}}
            st.success(f"Salvo '{nm}'. Reabra a sidebar para selecioná-lo.")

materials_all = list(MATERIALS.keys()) + list(st.session_state.CUSTOM_MATERIALS.keys())

# ---------- Plots básicos ----------
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Espectro (normalizado)")
    fig1, ax1 = plt.subplots()
    ax1.plot(E, spec)
    ax1.set_xlabel("Energia (keV)"); ax1.set_ylabel("Fluência relativa (1/keV)")
    ax1.set_title(f"Fonte: {spec_source}")
    st.pyplot(fig1)

with col2:
    st.markdown("#### μ(E) – Coeficiente de Atenuação Linear")
    mu_E = interpolate_mu(material, E, df_xcom)
    fig2, ax2 = plt.subplots()
    ax2.plot(E, mu_E, label=material)
    ax2.set_xlabel("Energia (keV)"); ax2.set_ylabel("μ (cm⁻¹)")
    ax2.legend(); st.pyplot(fig2)

# ---------- Mapa de Contraste ----------
st.markdown("---")
st.subheader("Mapa de Contraste (ΔI) por Espessura × Energia")

cm_col1, cm_col2, cm_col3 = st.columns(3)
ref_material = cm_col1.selectbox("Material de Referência (fundo)", materials_all, index=(materials_all.index("WATER") if "WATER" in materials_all else 0))
map_metric = cm_col2.selectbox("Métrica", ["ΔI = |I_mat − I_ref| (mono)", "Michelson = |I_A − I_B|/(I_A + I_B) (mono)"])
max_thick = cm_col3.slider("Espessura máxima (mm)", 1.0, 40.0, 20.0, 1.0)

E_grid = np.linspace(max(10.0, E.min()), float(kvp), 80)
T_grid = np.linspace(0.1, max_thick, 80)
TT = np.zeros((len(T_grid), len(E_grid)))

def mu_linear_for(mat_name, Eg):
    if mat_name in st.session_state.CUSTOM_MATERIALS:
        comp = st.session_state.CUSTOM_MATERIALS[mat_name]["fractions"]
        rho_c = st.session_state.CUSTOM_MATERIALS[mat_name]["density"]
        arr = np.zeros_like(Eg, dtype=float)
        for i, Ek in enumerate(Eg):
            s=0.0
            for el, w in comp.items():
                from core.xcom_builtin import elemental_mu_over_rho
                s += w * elemental_mu_over_rho(el, float(Ek))
            arr[i] = s*rho_c
        return arr
    else:
        return interpolate_mu(mat_name, Eg, None)

mu_mat = mu_linear_for(material, E_grid)
mu_ref = mu_linear_for(ref_material, E_grid)

for i, tmm in enumerate(T_grid):
    xcm = tmm/10.0
    I_mat = np.exp(-mu_mat * xcm)
    I_ref = np.exp(-mu_ref * xcm)
    if "Michelson" in map_metric:
        TT[i,:] = np.abs(I_mat - I_ref) / (I_mat + I_ref + 1e-12)
    else:
        TT[i,:] = np.abs(I_mat - I_ref)

fig_cm, ax_cm = plt.subplots()
im = ax_cm.imshow(TT, aspect='auto', origin='lower',
                  extent=[E_grid.min(), E_grid.max(), T_grid.min(), T_grid.max()])
ax_cm.set_xlabel("Energia (keV) — mono"); ax_cm.set_ylabel("Espessura (mm)")
ax_cm.set_title(f"{material} vs {ref_material}")
fig_cm.colorbar(im, ax=ax_cm, label="Contraste")

levels_default = [0.02, 0.05, 0.1, 0.2, 0.3]
levels_str = st.text_input("Níveis de isocontraste (separar por vírgula)",
                           value=", ".join([str(v) for v in levels_default]))
try:
    levels = [float(x.strip()) for x in levels_str.split(",") if x.strip()]
except Exception:
    levels = levels_default
Xg, Yg = np.meshgrid(E_grid, T_grid)
cs = ax_cm.contour(Xg, Yg, TT, levels=levels, colors='k', linewidths=0.8)
ax_cm.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
st.pyplot(fig_cm)

# ---------- Lookup pares separáveis ----------
st.subheader("Lookup de Pares de Materiais Mais Separáveis")
pair_metric = st.selectbox("Métrica integradora", ["Média de ΔI (mono)", "Média Michelson (mono)"], index=0)
n_top = st.slider("Quantos pares exibir (TOP N)", 3, 20, 5)
rows = []
for i in range(len(materials_all)):
    for j in range(i+1, len(materials_all)):
        A, B = materials_all[i], materials_all[j]
        muA = mu_linear_for(A, E_grid)
        muB = mu_linear_for(B, E_grid)
        scores = []; best = (0,0,0.0)
        for tmm in T_grid:
            xcm = tmm/10.0
            IA = np.exp(-muA*xcm); IB = np.exp(-muB*xcm)
            if "Michelson" in pair_metric: M = np.abs(IA-IB)/(IA+IB+1e-12)
            else: M = np.abs(IA-IB)
            scores.append(M.mean())
            jmax = int(np.argmax(M))
            if float(M[jmax]) > best[2]: best = (E_grid[jmax], tmm, float(M[jmax]))
        rows.append({"A":A, "B":B, "score":float(np.mean(scores)), "E_keV_max":best[0], "t_mm_max":best[1], "contraste_max":best[2]})
df_pairs = pd.DataFrame(rows).sort_values("score", ascending=False).head(n_top)
st.dataframe(df_pairs.reset_index(drop=True), use_container_width=True)

# ---------- Científico v8 ----------
st.markdown("---"); st.subheader("Propriedades Científicas Adicionais")
zcol1, zcol2, zcol3 = st.columns([1,1,1])
m_exp = zcol1.slider("Expoente m para Z_eff (Mayneord)", 2.0, 4.0, 2.94, 0.02)
fractions = get_fractions_for_material(material)
Zeff = compute_zeff(fractions, m_exp=m_exp)
zcol2.metric("Z_eff estimado", f"{Zeff:.2f}" if np.isfinite(Zeff) else "—")
rho_show = (st.session_state.CUSTOM_MATERIALS[material]["density"] if material in st.session_state.get("CUSTOM_MATERIALS", {}) else get_material(material).density_g_cm3)
zcol3.metric("Densidade (g/cm³)", f"{rho_show:.3f}")

st.markdown("#### μ/ρ (cm²/g) vs Energia (log–log)")
E_mu = np.linspace(10, 150, 300)
if material in st.session_state.get("CUSTOM_MATERIALS", {}):
    comp = st.session_state.CUSTOM_MATERIALS[material]["fractions"]
    mu_over_rho_curve = np.array([sum(comp[el]*elemental_mu_over_rho(el, float(Ek)) for el in comp) for Ek in E_mu])
else:
    from core.xcom_builtin import mix_mu_over_rho
    mu_over_rho_curve = mix_mu_over_rho(material, E_mu)
fig_mu, ax_mu = plt.subplots()
ax_mu.plot(E_mu, mu_over_rho_curve)
ax_mu.set_xscale("log"); ax_mu.set_yscale("log")
ax_mu.set_xlabel("Energia (keV)"); ax_mu.set_ylabel("μ/ρ (cm²/g)")
ax_mu.grid(True, which="both", alpha=0.3)
st.pyplot(fig_mu)

st.markdown("#### Beam hardening – Transmissão policromática vs espessura")
bh_cols = st.columns(3)
tmax_bh = bh_cols[0].slider("Espessura máx (mm)", 2.0, 50.0, 30.0, 1.0)
nsteps = bh_cols[1].slider("Passos", 20, 200, 80, 10)
mono_ref = bh_cols[2].selectbox("Mono de referência", ["E_efetiva (μ̄ ponderado)", "kVp/2", "pico (máx do espectro)"], index=0)
T_bh = np.linspace(0.0, tmax_bh, nsteps) # mm
muE = interpolate_mu(material, E, df_xcom)
I_poly = []
if mono_ref == "E_efetiva (μ̄ ponderado)":
    mu_eff_scalar = effective_mu(muE, spec, E); E_mono = None
elif mono_ref == "kVp/2":
    muE_interp = interp1d(E, muE, bounds_error=False, fill_value="extrapolate")
    E_mono = float(kvp)/2.0; mu_eff_scalar = float(muE_interp(E_mono))
else:
    idxp = int(np.argmax(spec)); E_mono = float(E[idxp]); mu_eff_scalar = float(muE[idxp])
for tmm in T_bh:
    xcm = tmm/10.0
    trans = np.exp(-muE * xcm)
    I_poly.append(float(np.trapz(spec * trans, E)))
I_poly = np.array(I_poly); I_poly /= (I_poly[0] if I_poly[0] > 0 else 1.0)
I_mono = np.exp(-(mu_eff_scalar) * (T_bh/10.0))
fig_bh, ax_bh = plt.subplots()
ax_bh.plot(T_bh, I_poly, label="Policromático (com endurecimento)")
ax_bh.plot(T_bh, I_mono, "--", label=f"Mono ref ({'E_eff' if E_mono is None else f'{E_mono:.1f} keV'})")
ax_bh.set_xlabel("Espessura (mm)"); ax_bh.set_ylabel("I/I0")
ax_bh.set_ylim(0,1.0); ax_bh.grid(alpha=0.3); ax_bh.legend()
st.pyplot(fig_bh)

st.markdown("---"); st.subheader("Comparação de Materiais – Overlay de μ(E)")
opts = materials_all
sel = st.multiselect("Selecione até 3 materiais", opts, default=[material], max_selections=3)
E_ol = np.linspace(max(10.0, E.min()), float(kvp), 250)
fig_ol, ax_ol = plt.subplots()
for mat in sel:
    if mat in st.session_state.get("CUSTOM_MATERIALS", {}):
        comp = st.session_state.CUSTOM_MATERIALS[mat]["fractions"]; rho_c = st.session_state.CUSTOM_MATERIALS[mat]["density"]
        mu_mat = np.array([sum(comp[el]*elemental_mu_over_rho(el, float(Ek)) for el in comp)*rho_c for Ek in E_ol])
    else:
        mu_mat = interpolate_mu(mat, E_ol, df_xcom)
    ax_ol.plot(E_ol, mu_mat, label=mat)
ax_ol.set_xlabel("Energia (keV)"); ax_ol.set_ylabel("μ (cm⁻¹)")
ax_ol.grid(alpha=0.3); ax_ol.legend(); st.pyplot(fig_ol)

st.markdown("---"); st.subheader("Tabela de Propriedades e Métricas")
options_compare = [m for m in materials_all if m != material]

# default só contendo valores realmente presentes nas opções
default_compare = []
if "WATER" in options_compare:
    default_compare = ["WATER"]

compare_to = st.multiselect(
    "Comparar contraste com",
    options_compare,
    default=default_compare
)

def mu_poly_eff(mat_name):
    if mat_name in st.session_state.get("CUSTOM_MATERIALS", {}):
        comp = st.session_state.CUSTOM_MATERIALS[mat_name]["fractions"]; rho_c = st.session_state.CUSTOM_MATERIALS[mat_name]["density"]
        mu = np.array([sum(comp[el]*elemental_mu_over_rho(el, float(Ek)) for el in comp)*rho_c for Ek in E])
    else:
        mu = interpolate_mu(mat_name, E, df_xcom)
    return effective_mu(mu, spec, E)

def fractions_of(mat_name):
    if mat_name in st.session_state.get("CUSTOM_MATERIALS", {}):
        return st.session_state.CUSTOM_MATERIALS[mat_name]["fractions"]
    from core.xcom_builtin import COMPOSITIONS
    return COMPOSITIONS.get(mat_name, {})

rows = []
main_mu_eff = mu_poly_eff(material)
rho_show = (st.session_state.CUSTOM_MATERIALS[material]["density"] if material in st.session_state.get("CUSTOM_MATERIALS", {}) else get_material(material).density_g_cm3)
rows.append({"Material": material, "Z_eff": round(compute_zeff(fractions_of(material), 2.94), 2), "ρ (g/cm³)": round(rho_show,3), "μ_eff (cm⁻¹)": round(main_mu_eff,4), "Contraste vs (ref)": "-", "Michelson (ref)": "-"})

xcm = thickness_mm/10.0
def I_poly_of(mat_name):
    if mat_name in st.session_state.get("CUSTOM_MATERIALS", {}):
        comp = st.session_state.CUSTOM_MATERIALS[mat_name]["fractions"]; rho_c = st.session_state.CUSTOM_MATERIALS[mat_name]["density"]
        mu_arr = np.array([sum(comp[el]*elemental_mu_over_rho(el, float(Ek)) for el in comp)*rho_c for Ek in E])
    else:
        mu_arr = interpolate_mu(mat_name, E, df_xcom)
    return float(np.trapz(spec * np.exp(-mu_arr * xcm), E))

I_main = I_poly_of(material)
for ref in compare_to:
    I_ref = I_poly_of(ref)
    contraste = abs(I_main - I_ref)
    michelson = contraste / (I_main + I_ref + 1e-12)
    rows.append({"Material": f"{material} vs {ref}", "Z_eff": "-", "ρ (g/cm³)": "-", "μ_eff (cm⁻¹)": round(mu_poly_eff(ref),4), "Contraste vs (ref)": round(contraste,4), "Michelson (ref)": round(michelson,4)})
df_props = pd.DataFrame(rows)
st.dataframe(df_props, use_container_width=True)
st.download_button("Baixar tabela (CSV)", data=df_props.to_csv(index=False).encode("utf-8"), file_name="propriedades_materiais.csv", mime="text/csv")


import streamlit as st
import numpy as np

st.markdown("---")
st.subheader("Exportar casos para Geant4 (GDML + macro)")

col1, col2, col3 = st.columns([1.2,1,1])
with col1:
    kvp_ui = st.number_input("Tensão do tubo (kVp)", 20.0, 200.0, 80.0, 1.0)
    prim_ui = st.number_input("Nº de primários", 1_000, 10_000_000, 200_000, 1_000)
with col2:
    al_ui = st.number_input("Filtração Al (mm)", 0.0, 10.0, 1.5, 0.1)
    cu_ui = st.number_input("Filtração Cu (mm)", 0.0, 2.0, 0.0, 0.01)
with col3:
    field_ui = st.number_input("Campo quadrado (mm)", 5.0, 200.0, 20.0, 1.0)
    gap_ui   = st.number_input("Gap amostra–detector (mm)", 0.1, 50.0, 1.0, 0.1)

use_spek = st.toggle("Usar SpekPy (se instalado)", value=True)
dump_flux = st.toggle("Exportar também fluxo (flux) no scoring", value=False)

# Lista de espessuras para varrer (mm)
thicks_str = st.text_input("Espessuras (mm), separadas por vírgula", "1, 5, 10, 20")
try:
    thicks = [float(x.strip()) for x in thicks_str.split(",") if x.strip() != ""]
except Exception:
    st.warning("Verifique a lista de espessuras. Ex.: 1, 5, 10, 20")
    thicks = []

# Nota para materiais CUSTOM_
if material.startswith("CUSTOM_"):
    st.info("Para CUSTOM_: o nome do material será usado como está no GDML. "
            "Se precisar de um material NIST específico, renomeie o CUSTOM_ para o nome Geant4 (ex.: G4_TISSUE_SOFT_ICRP).")

# Botão para gerar ZIP
if st.button("Gerar ZIP Geant4"):
    if not thicks:
        st.error("Defina pelo menos uma espessura válida.")
    else:
        bio = build_zip(
            material=material,
            thicknesses_mm=thicks,
            kvp=kvp_ui,
            filt_al=al_ui,
            filt_cu=cu_ui,
            use_spekpy=use_spek,
            primaries=int(prim_ui),
            field_mm=field_ui,
            det_gap_mm=gap_ui,
            include_flux=dump_flux,
        )
        st.download_button(
            "Baixar ZIP (GDML + macros)",
            data=bio.getvalue(),
            file_name=f"g4_cases_{material}.zip",
            mime="application/zip"
        )
        st.success("Pacote Geant4 gerado com sucesso!")
