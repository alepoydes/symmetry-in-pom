# symmetry-in-pom

Code and data for *Emergent Fourfold Symmetry in Polarization Images:
Local Origin and Diffractive Breakdown*. The figures can be rebuilt without
CUDA or TeX; saved optical calculations are supplied as a separate download.

Version 0.1.0, including code and both data archives, is preserved on
[Zenodo](https://doi.org/10.5281/zenodo.24323744).

## Rebuild the figures

Use the code and these four assets from the same
[GitHub release](https://github.com/alepoydes/symmetry-in-pom/releases/tag/v0.1.0)
or the [Zenodo archive](https://doi.org/10.5281/zenodo.24323744):

* `fourfold-source-data.zip` — 27 individual calibrated Jones fields, direct
  polarimetry frames, and their dark and reference measurements.
* `fourfold-derived-data.zip` — two relaxed director fields, calculated optical
  fields, processed measurements, analyses, display caches, and figure PDFs.
* `release-manifest.json` and `SHA256SUMS` — the matching code version and hashes.

Save the four assets together in a directory. On Linux, install Python 3.13,
`uv`, Make, `unzip`, Ghostscript (`gs`), and Poppler (`pdfinfo`), then run from
the code directory:

```bash
uv sync --locked --python 3.13
./scripts/unpack_release /path/to/release-directory
make -f reproduce.mk verify-results
make -f reproduce.mk figures
```

Install Computer Modern Unicode fonts (CMU Serif) to match the typography
of the published PDFs.

Unpacking checks the code snapshot and both archives before changing data.
`verify-results` checks 13 reported values at their quoted precision. The final
26 PDFs appear in `figs/`; [FIGURES.md](FIGURES.md) maps them to the article.
Lossless masters are saved in `build/figures/raw/`. To rebuild existing masters,
append `FORCE=1` to the figure command.

On a machine without a graphics device, install Mesa EGL (Debian/Ubuntu:
`libegl1` and `libgl1-mesa-dri`) and run
`LIBGL_ALWAYS_SOFTWARE=1 make -f reproduce.mk figures`.
The default Python environment contains no CUDA, CuPy, or Taichi dependency.

## Recompute numerical results

```bash
make -f reproduce.mk experiment  # process measured inputs; CPU
make -f reproduce.mk simulate    # optical calculations; NVIDIA GPU
make -f reproduce.mk analyze     # analyze measured and simulated fields; CPU
```

These stages skip complete, compatible outputs. `simulate` installs the optional
CUDA/Taichi dependencies. To inspect required work without running it, use
`uv run python code/scripts/reproduce_derived_analyses.py --dry-run`;
the experimental and simulated-optics drivers accept the same flag. Use the
printed command to rerun a single step. `FORCE=1` recomputes an entire stage.

## Starting data and director preparation

Experimental processing starts from individual calibrated Jones matrices;
camera holograms and their reconstruction are outside this package.

The director states were initialized with an analytical toron ansatz and
relaxed in lcsim. Parameters were adjusted manually in lcsim by comparing
TopoVec optical renders with experiment. The ansatz is described by
[Lobanov et al. (2022)](https://doi.org/10.3390/sym14122476).
There was no separate preparation script. The optical calculations use the
two supplied final states; their parameters and lcsim version are recorded in
`data_src/directors.toml`. The original discretized seeds and relaxation
trajectories are not included. The states retain their paths under
`data_src/directors/` but belong to the calculated-data archive.

## Prepare a matching code and data bundle

```bash
make -f reproduce.mk release-archives
make -f reproduce.mk export-code DEST=/path/to/new-directory
```

The first command writes the four assets to `build/release/`, binding the
archives to the version in `pyproject.toml` and SHA-256 hashes of the public
files. The data archives include only entries in `source-data-files.txt` and
`derived-data-files.txt`; diagnostic plots and other local files are excluded.
Append `FORCE=1` to replace an existing bundle. The second copies only
`public-files.txt` entries into a new directory, with `Makefile` and `README.md`
entry points. Neither command publishes anything. After changing public files
or data, prepare the matching bundle again.

## License and citation

The code, documentation, and accompanying data are distributed under the
[MIT license](LICENSE), included in both data archives. Third-party packages
retain their own licenses. Citation metadata and author credits are in
[CITATION.cff](CITATION.cff).
