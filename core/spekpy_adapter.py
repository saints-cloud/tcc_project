import numpy as np

def spekpy_available():
    try:
        import spekpy as sp  # noqa
        return True
    except Exception:
        return False

def _coerce_spectrum(spec_obj):
    if isinstance(spec_obj, (tuple, list)) and len(spec_obj) == 2:
        E_s, N_s = spec_obj
    elif isinstance(spec_obj, dict):
        for kE in ("E","energy","energies","keV"):
            if kE in spec_obj: 
                E_s = spec_obj[kE]; break
        else:
            raise TypeError("Chave de energia não encontrada no SpekPy.")
        for kN in ("N","fluence","spectrum","photons"):
            if kN in spec_obj:
                N_s = spec_obj[kN]; break
        else:
            raise TypeError("Chave de fluência não encontrada no SpekPy.")
    else:
        raise TypeError("Formato de retorno SpekPy não reconhecido.")
    E_s = np.asarray(E_s, dtype=float).ravel()
    N_s = np.asarray(N_s, dtype=float).ravel()
    if E_s.size != N_s.size:
        raise ValueError("E e N com tamanhos diferentes no SpekPy.")
    return E_s, N_s

def generate_spectrum_spekpy(kvp: float, energies_keV: np.ndarray,
                             filt_mm_Al: float = 1.5,
                             filt_mm_Cu: float = 0.0,
                             target: str = "W"):
    import inspect, numpy as np
    import spekpy as sp

    s = sp.Spek(kvp=float(kvp), th=12, targ=target)

    # filtros (algumas versões usam nomes ligeiramente diferentes)
    try:
        if filt_mm_Al > 0: s.filter("Al", float(filt_mm_Al))
        if filt_mm_Cu > 0: s.filter("Cu", float(filt_mm_Cu))
    except Exception:
        pass

    spec_obj = None
    errs = []

    # 1) tenta get_spectrum() sem kwargs (v2.5.x costuma aceitar)
    try:
        spec_obj = s.get_spectrum()
    except Exception as e:
        errs.append(f"get_spectrum(): {e}")

    # 2) tenta com diferentes nomes de parâmetros (emin/emax/ne ou n_bins)
    if spec_obj is None:
        try:
            sig = inspect.signature(s.get_spectrum)
            params = sig.parameters
            kwargs = {}
            if "emin" in params: kwargs["emin"] = 1.0
            if "e_min" in params: kwargs["e_min"] = 1.0
            if "emax" in params: kwargs["emax"] = float(kvp)
            if "e_max" in params: kwargs["e_max"] = float(kvp)
            if "ne"   in params: kwargs["ne"]   = 1000
            if "n"    in params: kwargs["n"]    = 1000
            if "n_bins" in params: kwargs["n_bins"] = 1000
            spec_obj = s.get_spectrum(**kwargs) if kwargs else None
        except Exception as e:
            errs.append(f"get_spectrum(kwargs): {e}")

    # 3) fallback para s.spectrum() (algumas releases expõem este nome)
    if spec_obj is None and hasattr(s, "spectrum"):
        try:
            spec_obj = s.spectrum()
        except Exception as e:
            errs.append(f"spectrum(): {e}")

    if spec_obj is None:
        raise RuntimeError("Falha SpekPy: " + " ; ".join(errs))

    # Extrai E e N de diferentes formatos
    E_s, N_s = _coerce_spectrum(spec_obj)

    # interpola para a grade pedida
    mask = np.isfinite(E_s) & np.isfinite(N_s)
    E_s, N_s = E_s[mask], N_s[mask]
    ord_ = np.argsort(E_s)
    E_s, N_s = E_s[ord_], N_s[ord_]

    N_interp = np.interp(energies_keV, E_s, N_s, left=0.0, right=0.0)
    area = float(np.trapz(N_interp, energies_keV))
    return N_interp/area if area > 0 else N_interp

