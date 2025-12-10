# core/g4_export.py
from io import BytesIO
import numpy as np
import zipfile
import os

# ——— util: tenta usar SpekPy; senão cai para TASMIP-like ———
def _spectrum(kvp, filt_al=1.5, filt_cu=0.0, use_spekpy=True):
    if use_spekpy:
        try:
            import spekpy as sp
            s = sp.Spek(kvp=float(kvp), th=12, targ="W")
            if filt_al > 0: s.filter("Al", float(filt_al))
            if filt_cu > 0: s.filter("Cu", float(filt_cu))
            try:
                spec = s.get_spectrum()
            except Exception:
                spec = s.spectrum()
            if isinstance(spec, dict):
                E = spec.get("E") or spec.get("energies") or spec.get("keV")
                N = spec.get("N") or spec.get("spectrum") or spec.get("photons")
            else:
                E, N = spec
            E = np.asarray(E, float).ravel()
            N = np.asarray(N, float).ravel()
            idx = np.argsort(E)
            E, N = E[idx], N[idx]
            area = float(np.trapz(N, E))
            if area > 0: N = N / area
            return E * 1e3, N  # retorna em keV
        except Exception:
            pass

    # fallback TASMIP-like (parabólico normalizado)
    E = np.linspace(10.0, float(kvp), 400)
    phi = E * (kvp - E)
    phi[phi < 0] = 0.0
    area = float(np.trapz(phi, E))
    phi = phi / area if area > 0 else phi
    return E * 1e3, phi  # em keV

# ——— mapeamento de materiais “lógicos” do app → nomes Geant4/NIST ———
def _g4_mat(name: str) -> str:
    n = name.upper()
    lut = {
        "WATER": "G4_WATER",
        "SOFT_TISSUE": "G4_TISSUE_SOFT_ICRP",
        "BONE_CORTICAL": "G4_BONE_COMPACT_ICRU",
        "BONE_TRABECULAR": "G4_B-100_BONE",  # fallback comum
        "PMMA": "G4_PLEXIGLASS",
        "DENTIN": "G4_HYDROXYAPATITE",
        "ENAMEL": "G4_HYDROXYAPATITE",
    }
    # Para CUSTOM_… deixa o próprio nome (usuário pode digitar um ref NIST válido)
    if n.startswith("CUSTOM_"):
        return n
    return lut.get(n, n)

def _write_gdml(material_name, thickness_mm, det_gap_mm, field_mm):
    mname = _g4_mat(material_name)
    slab_hz = thickness_mm / 2.0
    det_hz  = 0.05  # 0.1 mm total
    z_det   = slab_hz + det_gap_mm + det_hz
    world   = field_mm * 4
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<gdml>
  <materials>
    <material name="Vacuum"><D value="1e-25" unit="g/cm3"/></material>
  </materials>
  <solids>
    <box name="WorldBox" x="{world}mm" y="{world}mm" z="{world}mm"/>
    <box name="SlabBox"  x="{field_mm}mm"  y="{field_mm}mm"  z="{2*slab_hz}mm"/>
    <box name="DetBox"   x="{field_mm}mm"  y="{field_mm}mm"  z="{2*det_hz}mm"/>
  </solids>
  <structure>
    <volume name="World">
      <materialref ref="Vacuum"/>
      <solidref ref="WorldBox"/>
      <physvol><volumeref ref="Slab"/><position name="p1" x="0" y="0" z="0" unit="mm"/></physvol>
      <physvol><volumeref ref="Detector"/><position name="p2" x="0" y="0" z="{z_det}" unit="mm"/></physvol>
    </volume>
    <volume name="Slab">
      <materialref ref="{mname}"/>
      <solidref ref="SlabBox"/>
    </volume>
    <volume name="Detector">
      <materialref ref="G4_WATER"/>
      <solidref ref="DetBox"/>
    </volume>
  </structure>
  <setup name="Default" version="1.0"><world ref="World"/></setup>
</gdml>
"""

def _write_mac(geom_filename, kvp, n_prim, field_mm, E_keV=None, Phi=None, dump_flux=False):
    wmm = field_mm * 4.0
    lines = [
        "/control/verbose 1",
        "/run/verbose 1",
        f"/geometry/open GDML {os.path.basename(geom_filename)}",
        "/run/initialize",
        "",
        "# scoring",
        "/score/create/boxMesh m1",
        f"/score/mesh/boxSize {field_mm} {field_mm} 0.1 mm",
        "/score/mesh/nBin 1 1 1",
        "/score/quantity/energyDeposit edep",
    ]
    if dump_flux:
        lines += ["/score/quantity/flux flux"]
    lines += [
        "/score/close",
        "",
        "# GPS source",
        "/gps/pos/type Plane",
        "/gps/pos/shape Square",
        f"/gps/pos/centre 0 0 -{wmm/2 - 1.0} mm",
        f"/gps/pos/halfx {field_mm/2} mm",
        f"/gps/pos/halfy {field_mm/2} mm",
        "/gps/ang/type beam1",
        "/gps/ang/maxtheta 0 deg",
    ]
    if E_keV is not None and Phi is not None and len(E_keV) == len(Phi):
        # espectro arbitrário
        area = float(np.trapz(Phi, E_keV))
        W = Phi / area if area > 0 else Phi
        lines += ["/gps/ene/type Arb", "/gps/hist/type energy"]
        for e, w in zip(E_keV, W):
            lines += [f"/gps/hist/point {e/1000.0:.6f} {float(w):.8f}  # {e:.1f} keV"]
    else:
        # mono de fallback
        lines += ["/gps/ene/type Mono", f"/gps/ene/mono {kvp/2/1000.0:.6f} MeV"]
    lines += ["", f"/run/beamOn {int(n_prim)}", "/score/dumpQuantityToFile m1 edep m1_edep.csv"]
    if dump_flux:
        lines += ["/score/dumpQuantityToFile m1 flux m1_flux.csv"]
    return "\n".join(lines)

def build_zip(material, thicknesses_mm, kvp, filt_al, filt_cu, use_spekpy, primaries,
              field_mm=20.0, det_gap_mm=1.0, include_flux=False) -> BytesIO:
    E, Phi = _spectrum(kvp, filt_al, filt_cu, use_spekpy=use_spekpy)
    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as z:
        # README
        z.writestr("README.txt",
                   "Casos Geant4 gerados pelo app (GDML + macros). Execute cada run_*.mac no seu executável.\n")
        for t in thicknesses_mm:
            case = f"{material}_{t:.1f}mm"
            folder = case + "/"
            geom_name = f"geom_{case}.gdml"
            mac_name  = f"run_{case}.mac"
            # arquivos
            z.writestr(folder + geom_name, _write_gdml(material, float(t), float(det_gap_mm), float(field_mm)))
            z.writestr(folder + mac_name,
                       _write_mac(geom_name, float(kvp), int(primaries), float(field_mm),
                                  E_keV=E, Phi=Phi, dump_flux=include_flux))
    bio.seek(0)
    return bio
