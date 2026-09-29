import os
def report(ctx):
    return {"x": open("/etc/passwd").read()[:10]}
