# ARC-3 local dependency scope, 2026-10-06

No packages, games or models were installed or downloaded by this inspection.
Only official documentation/package metadata was read. No credentials, environment
secrets, remote jobs or security settings were accessed or changed.

The minimum supported route for the existing `arc3_env.py` harness is the official
`arc-agi` toolkit. The inspected published release is 0.9.9 and requires Python
3.12 or newer. The bundled local Python 3.12.14 already meets that requirement.
Use an isolated repository-local environment rather than modifying Codex's bundled
runtime. These are proposed commands, not commands that were executed:

```powershell
& 'C:\Users\wkeif\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m venv .venv-arc3
& '.\.venv-arc3\Scripts\python.exe' -m pip install 'arc-agi==0.9.9'
```

Sources: [official toolkit quickstart](https://docs.arcprize.org/toolkit/overview)
and [publisher package metadata](https://pypi.org/project/arc-agi/).

The toolkit's direct dependency scope is `arcengine>=0.9.3`, Flask, Matplotlib,
Pydantic, python-dotenv, Pillow and requests. Dependencies bring their own
transitive packages; their resolved versions and total bytes have not yet been
measured or locked. An isolated installation should retain its resolver report
and final versions. The official dependency definition is
[arcprize/ARC-AGI pyproject.toml](https://github.com/arcprize/ARC-AGI/blob/main/pyproject.toml).
The engine source is [arcprize/ARCEngine](https://github.com/arcprize/ARCEngine).

For this project's offline experiment, specify `OperationMode.OFFLINE` and an
explicit environment directory. Offline operation needs no API key. Normal/online
mode can fetch an anonymous key, so the generic `Arcade()` quickstart is not the
appropriate launch command for this scope. Local game metadata/source files are
a separate requirement; the package installation alone does not establish that
ar25/lp85/sc25 are present. Sources: [local versus online](https://docs.arcprize.org/local-vs-online)
and the [toolkit constructor reference](https://pypi.org/project/arc-agi/).

## Authorization and blockers

Ordinary repository-local Python dependency installation fits the recorded owner
authorization for necessary desktop runtime preparation. This is the current
scope interpretation, not a new permission request. The parent asked for the
installation path only in this turn, so no install was performed. That permission
does not expand the explicit no-Hetzner boundary or authorize new accounts,
credentials, online API calls, paid services, OS/security changes or benchmark
evaluation access. Public game acquisition needs its source and scope identified;
do not silently fetch games by switching to the toolkit's network-enabled mode.

Current concrete blockers:

- `arc_agi` and `arcengine` are absent from the Python used for local work.
- Local game files and the required versioned game metadata are unavailable;
  `/srv/arc3-claude/environment_files` is only a documented server location.
- The existing model-written policy path lacks a bounded-memory sandbox on
  Windows; WSL availability was not established and Docker was not found. Package
  installation does not resolve this policy-execution boundary.
- A trusted-engine, data-only action baseline could avoid arbitrary authored
  policy execution once packages and authorized game files are available. Keep
  it separate from policy-tensor evolution, with frozen action/time limits and
  complete observation/action receipts.
- Current remote jobs and server package versions remain unobserved because
  Hetzner work is excluded. No secret/authentication diagnosis is needed for
  the offline local route.

The three ARC-AGI-2 direct-prediction packets are already frozen separately at
`runs/weights-native-arc-direct-batch02-20261006/`; ARC-3 setup is not a dependency
of that batch. Carry the previously red/incomplete full standard status; no
unrelated full suites were rerun for this preparation.
