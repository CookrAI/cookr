"""Tiny RunPod GraphQL client: rent, inspect, stop, kill a pod. No SDK.
Key from RUNPOD_API_KEY or ./.env.

  python train/runpod_api.py create [--gpu "NVIDIA GeForce RTX 4090"] [--cloud COMMUNITY]
  python train/runpod_api.py status <podId>      -> ssh host/port once the runtime is up
  python train/runpod_api.py stop <podId>        -> keeps the volume, stops billing the GPU
  python train/runpod_api.py terminate <podId>   -> deletes everything
  python train/runpod_api.py list
"""
import argparse
import json
import os
import pathlib
import sys
import time

import httpx

ROOT = pathlib.Path(__file__).resolve().parent.parent
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


def key() -> str:
    k = os.environ.get("RUNPOD_API_KEY")
    if not k and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith("RUNPOD_API_KEY="):
                k = line.split("=", 1)[1].strip()
    if not k:
        sys.exit("RUNPOD_API_KEY missing")
    return k


def gql(query: str, variables: dict | None = None) -> dict:
    r = httpx.post("https://api.runpod.io/graphql", headers={"Authorization": f"Bearer {key()}"},
                   json={"query": query, "variables": variables or {}}, timeout=60)
    r.raise_for_status()
    d = r.json()
    if d.get("errors"):
        raise RuntimeError(json.dumps(d["errors"]))
    return d["data"]


def create(a):
    pub = pathlib.Path(a.pubkey).expanduser().read_text().strip()
    q = """mutation($in: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $in) { id imageName machineId costPerHr } }"""
    inp = {"cloudType": a.cloud, "gpuCount": 1, "gpuTypeId": a.gpu, "name": a.name, "imageName": IMAGE,
           "volumeInGb": a.volume, "containerDiskInGb": a.disk, "volumeMountPath": "/workspace",
           "minVcpuCount": 4, "minMemoryInGb": 24, "ports": "22/tcp", "dockerArgs": "",
           "env": [{"key": "PUBLIC_KEY", "value": pub}]}
    d = gql(q, {"in": inp})["podFindAndDeployOnDemand"]
    print(json.dumps(d))


def status(a):
    q = """query($id: String!) { pod(input:{podId:$id}) { id name desiredStatus costPerHr gpuCount machine { gpuDisplayName } runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } } }"""
    d = gql(q, {"id": a.pod})["pod"]
    ssh = None
    for p in ((d.get("runtime") or {}).get("ports") or []):
        if p["privatePort"] == 22 and p["isIpPublic"]:
            ssh = f"{p['ip']}:{p['publicPort']}"
    print(json.dumps({"id": d["id"], "status": d["desiredStatus"], "gpu": (d.get("machine") or {}).get("gpuDisplayName"),
                      "costPerHr": d["costPerHr"], "uptime": (d.get("runtime") or {}).get("uptimeInSeconds"), "ssh": ssh}))


def stop(a):
    print(gql("""mutation($id: String!) { podStop(input:{podId:$id}) { id desiredStatus } }""", {"id": a.pod}))


def terminate(a):
    print(gql("""mutation($id: String!) { podTerminate(input:{podId:$id}) }""", {"id": a.pod}))


def list_(a):
    d = gql("""{ myself { clientBalance currentSpendPerHr pods { id name desiredStatus costPerHr machine { gpuDisplayName } } } }""")["myself"]
    print("balance", round(d["clientBalance"], 2), "spend/hr", d["currentSpendPerHr"])
    for p in d["pods"]:
        print(p["id"], p["name"], p["desiredStatus"], (p.get("machine") or {}).get("gpuDisplayName"), p["costPerHr"])


def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    c = s.add_parser("create"); c.add_argument("--gpu", default="NVIDIA GeForce RTX 4090"); c.add_argument("--cloud", default="COMMUNITY")
    c.add_argument("--name", default="cookr-train"); c.add_argument("--volume", type=int, default=80); c.add_argument("--disk", type=int, default=40)
    c.add_argument("--pubkey", default="~/.ssh/cookr_runpod.pub"); c.set_defaults(fn=create)
    for n, fn in (("status", status), ("stop", stop), ("terminate", terminate)):
        x = s.add_parser(n); x.add_argument("pod"); x.set_defaults(fn=fn)
    s.add_parser("list").set_defaults(fn=list_)
    a = p.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
