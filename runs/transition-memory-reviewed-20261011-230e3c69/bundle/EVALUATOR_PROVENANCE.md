# Evaluator provenance correction

Research source:
 /srv/cellforge-inbox/research-20261008/source/scripts/arc3_env.py
Deployed first study:
 /srv/cellforge-inbox/continuous-spatial-20261011/arc3_env.py
Next review copy:
 arc3_env.py
All three chosen copies have frozen SHA256
958e922ed38e426f2004b1cb1afaf128e1d9b6c8a5046095afdb3d194e689ca0.

Compared with /srv/cellforge-inbox/head-20261001/arc3_env.py, after normalizing
line endings, the research source contains four additional lines:

```diff
-    prev = last_frame(resp)
+    prev = last_frame(resp) if resp is not None else None
+    if resp is None:
+        out["error"] = "environment returned no initial observation"
...
             if resp is None:
+                out["error"] = "environment returned no observation after reset"
...
         if resp is None:
+            out["error"] = "environment returned no observation after action"
```

There are no other normalized text changes in this bounded comparison.
These report missing observations as explicit errors; they do not change the
completed-level scoring rules. The earlier byte-identical wording does not apply
to historical head. This review copy introduces no additional evaluator edits.
Initial-state/legal-mask instrumentation in worker.py wraps E.new_env/C.Policy
and returns their original values. It never supplies altered frames to the
environment or changes the scoring/RESET/action allowance.

