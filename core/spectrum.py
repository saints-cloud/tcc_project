import numpy as np

def w_target_spectrum_tasmip_like(kvp: float, energies_keV, filt_mm_Al: float = 1.5, filt_mm_Cu: float = 0.0):
    E = np.asarray(energies_keV, dtype=float)
    # forma simples tipo TASMIP: φ(E) ~ E*(Emax-E) para 0<E<Emax
    Emax = float(kvp)
    phi = np.clip(E * (Emax - E), 0, None)
    # filtragem simples exponencial com μ/ρ efetivo aproximado
    # (valores heurísticos para ilustração, SpekPy é recomendado para realismo)
    mu_rho_Al = 0.15  # cm^2/g @ ~50 keV (aprox)
    mu_rho_Cu = 0.45
    rho_Al = 2.70; rho_Cu = 8.96
    t_Al_cm = (filt_mm_Al/10.0)
    t_Cu_cm = (filt_mm_Cu/10.0)
    # dependência ~ 1/E para filtragem (heurística simples)
    att = np.exp(-(mu_rho_Al*rho_Al*t_Al_cm + mu_rho_Cu*rho_Cu*t_Cu_cm) * (50.0/np.clip(E,1,1e9)))
    phi = phi * att
    area = float(np.trapz(phi, E))
    return phi/area if area>0 else phi
