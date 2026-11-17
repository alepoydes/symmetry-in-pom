# Calculated results and rendering data

This tree contains registered experimental averages, HDR frames, simulated
Jones fields, analysis reports and arrays, and prepared display panels.
The processing and figure scripts regenerate these outputs from the
experimental measurements and the two saved director states.

`fourfold-derived-data.zip` contains the files listed in `derived-data-files.txt`:
the required numerical products and display caches, the final PDFs in `figs/`,
and the numerical director
checkpoints at their existing `data_src/directors/` paths, with
`data_src/directors.toml` and `data_src/directors.sha256`. The checkpoints
cannot be reconstructed exactly from the available relaxation records and
must be preserved. They are calculated states, not experimental data.

The calculated data and figure PDFs are distributed under the MIT license in
`LICENSE`, also included in each release archive.

The artifact permits figure generation without repeating the optical solvers.
See `REPRODUCTION.md` for CPU rendering and selective recomputation. Nothing
under `data/` is required in Git apart from this description.
