import numpy as np

# Elemental mass attenuation (cm^2/g) at discrete energies (keV)
XCOM_ELEMENTAL = {
    "H": {40.0: 0.3441, 50.0: 0.3344, 60.0: 0.3253, 80.0: 0.3087, 100.0: 0.2941, 150.0: 0.2650, 200.0: 0.2428, 300.0: 0.2112, 400.0: 0.1893},
    "C": {40.0: 0.02029, 50.0: 0.01806, 60.0: 0.01693, 80.0: 0.01566, 100.0: 0.01482, 150.0: 0.01331, 200.0: 0.01220, 300.0: 0.01063, 400.0: 0.009526},
    "N": {40.0: 0.04254, 50.0: 0.02867, 60.0: 0.02263, 80.0: 0.01770, 100.0: 0.01566, 150.0: 0.01338, 200.0: 0.01213, 300.0: 0.01051, 400.0: 0.009412},
    "O": {40.0: 0.08918, 50.0: 0.05191, 60.0: 0.03573, 80.0: 0.02299, 100.0: 0.01831, 150.0: 0.01418, 200.0: 0.01251, 300.0: 0.01068, 400.0: 0.009535},
    "Na": {40.0: 1.872, 50.0: 1.734, 60.0: 1.655, 80.0: 1.553, 100.0: 1.476, 150.0: 1.330, 200.0: 1.220, 300.0: 1.062, 400.0: 0.09523},
    "Mg": {40.0: 0.07250, 50.0: 0.04337, 60.0: 0.03072, 80.0: 0.02073, 100.0: 0.01699, 150.0: 0.01355, 200.0: 0.01206, 300.0: 0.01035, 400.0: 0.009245},
    "P":  {40.0: 0.03437, 50.0: 0.02443, 60.0: 0.02008, 80.0: 0.01643, 100.0: 0.01484, 150.0: 0.01289, 200.0: 0.01173, 300.0: 0.01018, 400.0: 0.009118},
    "S":  {40.0: 0.02394, 50.0: 0.01792, 60.0: 0.01486, 80.0: 0.01222, 100.0: 0.01102, 150.0: 0.009566, 200.0: 0.008670, 300.0: 0.007533, 400.0: 0.006728},
    "Cl": {40.0: 0.07857, 50.0: 0.05875, 60.0: 0.04532, 80.0: 0.02889, 100.0: 0.02104, 150.0: 0.01469, 200.0: 0.01253, 300.0: 0.01050, 400.0: 0.009327},
    "K":  {40.0: 1.422, 50.0: 0.7857, 60.0: 0.5875, 80.0: 0.3251, 100.0: 0.2345, 150.0: 0.1582, 200.0: 0.1319, 300.0: 0.1080, 400.0: 0.09495},
    "F":  {40.0: 0.02454, 50.0: 0.01962, 60.0: 0.01738, 80.0: 0.01533, 100.0: 0.01426, 150.0: 0.01266, 200.0: 0.01158, 300.0: 0.01007, 400.0: 0.009027},
    "Ca": {40.0: 1.700, 50.0: 0.09291, 60.0: 0.05914, 80.0: 0.03254, 100.0: 0.02304, 150.0: 0.01548, 200.0: 0.01303, 300.0: 0.01083, 400.0: 0.009596},
}

MATERIAL_DENSITY = {
    "WATER": 1.0,
    "SOFT_TISSUE": 1.06,
    "BONE_CORTICAL": 1.85,
    "BONE_TRABECULAR": 1.20,
    "ENAMEL": 2.90,
    "DENTIN": 2.10,
    "PMMA": 1.19,
}

# mass fractions per material (approx ICRU/ICRP + literature)
COMPOSITIONS = {
    "WATER": {"H":0.1119,"O":0.8881},
    "SOFT_TISSUE": {"H":0.101,"C":0.111,"N":0.026,"O":0.762},
    "BONE_CORTICAL": {"H":0.034,"C":0.155,"N":0.042,"O":0.435,"Na":0.001,"Mg":0.002,"P":0.103,"S":0.003,"Cl":0.002,"K":0.001,"Ca":0.122},
    "BONE_TRABECULAR": {"H":0.060,"C":0.210,"N":0.040,"O":0.570,"Na":0.002,"Mg":0.002,"P":0.045,"S":0.003,"Cl":0.002,"K":0.001,"Ca":0.065},
    "ENAMEL": {"H":0.024,"C":0.003,"N":0.0,"O":0.437,"Mg":0.003,"P":0.178,"Ca":0.355},
    "DENTIN": {"H":0.041,"C":0.151,"N":0.042,"O":0.455,"Mg":0.001,"P":0.103,"Ca":0.207},
    "PMMA": {"H":0.0806,"C":0.6006,"O":0.3188},
}

def _interp_loglog(xp, yp, x):
    """
    Interpola em log–log e faz extrapolação linear em log abaixo/acima da faixa.
    Aceita x escalar ou array. Retorna escalar se a entrada for escalar.
    """
    xp = np.asarray(list(xp), dtype=float)
    yp = np.asarray(list(yp), dtype=float)
    # garante pelo menos 1D para evitar array 0-D
    x  = np.atleast_1d(np.asarray(x, dtype=float))

    lx  = np.log(xp)
    ly  = np.log(yp)
    lxi = np.log(x)

    # interpola dentro da faixa
    y = np.interp(lxi, lx, ly)

    # extrapolação à esquerda (usa inclinação dos dois primeiros pontos)
    if lxi.min() < lx[0]:
        mL = (ly[1] - ly[0]) / (lx[1] - lx[0])
        left = lxi < lx[0]
        y[left] = ly[0] + mL * (lxi[left] - lx[0])

    # extrapolação à direita (usa inclinação dos dois últimos pontos)
    if lxi.max() > lx[-1]:
        mR = (ly[-1] - ly[-2]) / (lx[-1] - lx[-2])
        right = lxi > lx[-1]
        y[right] = ly[-1] + mR * (lxi[right] - lx[-1])

    y = np.exp(y)
    # devolve escalar se a entrada era escalar
    return float(y[0]) if y.size == 1 else y



def elemental_mu_over_rho(element: str, energy_keV: float) -> float:
    tbl = XCOM_ELEMENTAL[element]
    energies = np.array(sorted(tbl.keys()), dtype=float)
    vals = np.array([tbl[e] for e in energies], dtype=float)
    return float(_interp_loglog(energies, vals, energy_keV))

def mix_mu_over_rho(material: str, energies_keV: np.ndarray) -> np.ndarray:
    mat = material.upper()
    comp = COMPOSITIONS.get(mat, {})
    out = np.zeros_like(energies_keV, dtype=float)
    for i, E in enumerate(energies_keV):
        s = 0.0
        for el, w in comp.items():
            s += w * elemental_mu_over_rho(el, float(E))
        out[i] = s
    return out
