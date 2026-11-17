# Experimental inputs and numerical checkpoints

The experimental artifact contains:

* `holography/{450,520,660}/{1,...,9}.npz`: all 27 individually reconstructed,
  calibrated Jones matrices, before registration and coherent averaging.
* `direct_pom/520nm/Exposure_{12,42}ms/`: frames `0045.png` through `0049.png`,
  `Environment.png` (dark measurement), and nine `Ground` reference images
  per exposure. Both exposures are needed for HDR combination.
* `experiment.toml`: physical sampling, wavelength conventions, acquisition
  settings, and calibration information.
* `checksums.sha256`: hashes of this experimental input set.

The experimental artifact contains neither camera holograms nor the earlier
holographic reconstruction procedure. Each wavelength contains measurements of
nine distinct neighboring relaxed torons under common conditions. The numerical
analysis starts from these individually calibrated Jones matrices.

The measured centers of the primary input sources are 518.2 +/- 0.5 nm and
522.5 +/- 0.5 nm. The reported propagation control uses fixed values of
518.2 and 522 nm against the common 520-nm calculation; these numerical inputs
are recorded separately in `holography.column_wavelength_control`.

Two saved numerical director fields retain their existing paths under
`directors/` to preserve the provenance of already calculated optical fields.
They and `directors.toml` are distributed in the **derived-data artifact**,
not the experimental artifact. `directors.sha256` verifies this separate group.
The states were prepared from an analytical toron ansatz and relaxed in lcsim;
parameters were adjusted manually against the experimental optics using TopoVec
renders within lcsim. See [Lobanov et al. (2022)](https://doi.org/10.3390/sym14122476)
for the analytical ansatz. There was no separate preparation script. The saved
states and `directors.toml` provide the inputs for reproducing the optical
calculations; the original discretized seeds and relaxation trajectories are
not included.

These data are distributed under the MIT license in `LICENSE`, also included
in each release archive.

Large files are excluded from Git; the TOML descriptions and SHA-256 lists are
tracked. Use `scripts/check_data_src` for experimental inputs and
`scripts/check_directors` for the numerical checkpoints. Intentional changes
require updating the corresponding list with
`scripts/update_data_src_checksums --force` or the same command with `--directors`.
