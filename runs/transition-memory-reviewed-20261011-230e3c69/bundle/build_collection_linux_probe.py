import base64,json,pathlib
root=pathlib.Path(".")
payload={"files":{n:(root/n).read_text() for n in ("common.py","supervisor.py")},
 "plan":json.loads((root/"PLAN.json").read_text()),
 "worker":(root/"collection_fixture_worker.py").read_text(),
 "driver":(root/"collection_fixture_driver.py").read_text()}
encoded=base64.b64encode(json.dumps(payload,separators=(",",":")).encode()).decode()
template=(root/"collection_linux_template.py").read_text()
(root/"collection_linux_stream.py").write_text(template.replace("PAYLOAD",repr(encoded)),encoding="utf-8")
print("Prepared isolated real failure/backoff/collection crash probes")

