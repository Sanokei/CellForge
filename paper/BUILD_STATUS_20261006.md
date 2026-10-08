# Dated manuscript PDF: blocked, 6 October 2026

No current PDF was produced. The intended distinct output,
`paper/cellforge-20261006.pdf`, does not exist. Its page count, revised content
and visual layout therefore cannot be verified. No historical PDF was renamed
or presented as the revised manuscript.

The source at this recorded build attempt was [cellforge.tex](cellforge.tex), SHA256
`a0bc9fec7d5a790dbdf1f2d1f2b42ac6420aa44dfab0f0aff639a4791a701451`.
The subsequent direct-ARC baseline documentation update changed that source;
its current hash is in [validation-20261006.json](validation-20261006.json).
No compiler or PDF verification was rerun for that later update. This report
and its JSON receipts retain the actual earlier build-attempt state.

## Exact observed blockers

1. The original and follow-up built-in compile invocation targeted
   `D:\Github\CellForge\paper\cellforge.tex`. Both returned `isError: true`
   with: **"LaTeX compilation did not complete. Compilation status is unknown;
   do not change the source based on this result."** No compiler diagnostics,
   process exit code, exported PDF or auxiliary build files were returned.
   The tool does not expose the cause. Its documented interface returns
   diagnostics rather than exporting a PDF. The follow-up response and exact
   invocation are preserved in [compiler-result-20261006.json](compiler-result-20261006.json).
2. No `pdflatex`, `lualatex`, `xelatex`, `latexmk`, `tectonic` or `tex` compiler
   was found on PATH. Standard TeX Live/MiKTeX locations, user Scoop locations,
   installed-program registry entries, the bundled runtime and the installed
   PDF plugin supplied no usable compiler. The runtime's native bundles are
   Git, jxrlib, libheif, Poppler and PowerShell.
3. `wsl --list --quiet` returned installation/usage text rather than an installed
   distribution. A bounded `wsl --status` exited with code **50** and no output.
   No working WSL compiler route was established. No installation or subsystem
   configuration was attempted.

Shell/Python/Poppler process access worked. This is not a demonstrated general
filesystem or process denial. The unavailable compiler/export path is the
build blocker; the built-in compiler's internal failure cause is unreported.
Evidence: [build-attempt-20261006.json](build-attempt-20261006.json).

## Preservation and verification

The three historical PDFs are byte-identical to their tracked HEAD versions:

| File | SHA256 |
| --- | --- |
| `cellforge.pdf` | `d69cd802f46e2f95131bcfffd5382228b2fcb433369886694a530b57a32acb59` |
| `cellforge_clean.pdf` | `d69cd802f46e2f95131bcfffd5382228b2fcb433369886694a530b57a32acb59` |
| `cellforge_pre_cleanup.pdf` | `c8ff25f884130d7bb406e677da092a8a38170c8fc4ab4e29acc36516dd2d74d5` |

Poppler and pypdf agree that historical `cellforge.pdf` has **16 pages**.
Its PDF metadata dates the build to 30 September 2026. Extracted text lacks
the new native-study heading, the 6 October draft date and the identity-router
composition heading. Those observations confirm that it is not the revised
paper; they do not validate a current PDF.

Only this build report, its JSON evidence and the revision-note pointer changed
in this follow-up. The manuscript source, historical PDFs, implementation,
experiments, tests and floors were unchanged. No standard gates were rerun.
No installs, downloads, server access, submissions or publication occurred.

To finish, the parent needs a working already-installed LaTeX compiler with
file export, or a functioning supported editor PDF-export route. Build to the
distinct dated filename, then verify page count, updated text and rendered
pages. A synthetic re-typesetting of the source or a renamed historical PDF
would not close this compilation requirement.
