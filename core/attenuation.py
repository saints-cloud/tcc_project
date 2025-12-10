import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from .xcom_builtin import mix_mu_over_rho
from .materials import get_material

def _mu_from_csv(material: str, energy_keV: np.ndarray, df: pd.DataFrame | None):
    if df is None: 
        return None
    m = material.lower()
    if m not in df.columns:
        return None
    # assume primeira coluna é energia keV
    Ecol = df.columns[0]
    E_tab = np.asarray(df[Ecol].values, dtype=float)
    mu_tab = np.asarray(df[m].values, dtype=float)
    f = interp1d(E_tab, mu_tab, bounds_error=False, fill_value="extrapolate")
    return f(energy_keV)

def interpolate_mu(material: str, energy_keV: np.ndarray, df_xcom: pd.DataFrame | None) -> np.ndarray:
    # tenta CSV do usuário
    mu_csv = _mu_from_csv(material, energy_keV, df_xcom)
    if mu_csv is not None:
        return mu_csv
    # regra de mistura embutida + densidade
    E_hi = np.linspace(10, 400, 300)
    mu_over_rho = mix_mu_over_rho(material, E_hi)
    rho = get_material(material).density_g_cm3
    # interpola log–log para suavidade
    f = interp1d(np.log(E_hi), np.log(mu_over_rho * rho), bounds_error=False, fill_value="extrapolate")
    return np.exp(f(np.log(energy_keV)))

def effective_mu(mu_E: np.ndarray, spectrum: np.ndarray, energies_keV: np.ndarray) -> float:
    # μ_eff = <μ>_spec = ∫ μ(E) φ(E) dE / ∫ φ(E) dE
    num = float(np.trapz(mu_E * spectrum, energies_keV))
    den = float(np.trapz(spectrum, energies_keV))
    return num / den if den > 0 else 0.0
