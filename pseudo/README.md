# Pseudopotentials

This directory stores UPF pseudopotential files used by DFT-Zig.
The files are **not** included in the repository. Download them before running calculations.

## Required Files

| File | Source | Notes |
|------|--------|-------|
| `Si.upf` | SG15 ONCV v1.1 | `Si_ONCV_PBE-1.1.upf` renamed |
| `C.upf` | SG15 ONCV v1.0 | `C_ONCV_PBE-1.0.upf` renamed |
| `Al.upf` | SG15 ONCV v1.0 (3e) | `Al_ONCV_PBE-1.0.upf` renamed |
| `Al_3e.upf` | Same as `Al.upf` | 3-electron Al |
| `Al_11e.upf` | SG15 ONCV v1.2 (11e) | `Al_ONCV_PBE-1.2.upf` renamed |

## Download

### SG15 ONCV Library (CC BY-SA 4.0)

Download from: http://www.quantum-simulation.org/potentials/sg15_oncv/

```sh
# Example: download Si
curl -O http://www.quantum-simulation.org/potentials/sg15_oncv/upf/Si_ONCV_PBE-1.1.upf
cp Si_ONCV_PBE-1.1.upf Si.upf
```

### PseudoDojo (CC BY 4.0)

Alternative source: http://www.pseudo-dojo.org/

## PAW Pseudopotentials

For PAW calculations, download from pslibrary:

| File | Source |
|------|--------|
| `Si.pbe-n-kjpaw_psl.1.0.0.UPF` | [pslibrary](https://dalcorso.github.io/pslibrary/) |
| `Fe.pbe-spn-kjpaw_psl.0.2.1.UPF` | [pslibrary](https://dalcorso.github.io/pslibrary/) |

## License

- **SG15 ONCV**: CC BY-SA 4.0. Cite: Schlipf & Gygi, CPC 196, 36 (2015); Hamann, PRB 88, 085117 (2013)
- **pslibrary**: Cite: Dal Corso, CMS 95, 337 (2014)
