import base64,json,pathlib
root=pathlib.Path(".")
files={n:(root/n).read_text() for n in ("common.py","supervisor.py")}
payload={"files":files,"plan":json.loads((root/"PLAN.json").read_text()),
 "fixture_worker":(root/"linux_fixture_worker.py").read_text(),
 "fixture_driver":(root/"linux_fixture_driver.py").read_text()}
encoded=base64.b64encode(json.dumps(payload,separators=(",",":")).encode()).decode()
template=(root/"linux_probe_template.py").read_text()
(root/"linux_probe_stream.py").write_text(template.replace("PAYLOAD",repr(encoded)),encoding="utf-8")
print("Prepared bounded isolated Linux operational probes")

