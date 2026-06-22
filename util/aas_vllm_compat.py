"""AAS vLLM compatibility shim (auto-applied for every interpreter in .venv).

Problem
-------
vLLM 0.11.0 mounts a prometheus-fastapi-instrumentator metrics middleware
unconditionally (api_server.py: mount_metrics -> Instrumentator().instrument()).
That middleware walks `app.routes` via `prometheus_fastapi_instrumentator.routing.
_get_route_name`, which does a bare `route.path`. Under the bleeding-edge web
stack installed here (starlette 1.3.x / fastapi 0.137.x), `app.routes` contains
an `_IncludedRouter` object that has no `.path`, so EVERY request — including the
`GET /v1/models` readiness probe — 500s with:

    AttributeError: '_IncludedRouter' object has no attribute 'path'

and the server never becomes ready. instrumentator 8.0.0 is the newest release
and still has this bug, so there is no version to upgrade to.

Fix
---
Replace `routing._get_route_name` with a tolerant version that uses
`getattr(route, "path", ...)` and recurses into any route exposing sub-`routes`
(covers Mount and starlette-1.3 `_IncludedRouter`). `routing.get_route_name`
calls `_get_route_name` by module global, so this single reassignment fixes the
whole metrics path. Behaviour is unchanged for routes that already have `.path`;
metric transaction labels remain correct.

Delivery
--------
Installed into site-packages and auto-imported via `aas_vllm_compat.pth`, so it
applies to all vLLM launch paths (start_vllm.py AND the direct
`python -m vllm.entrypoints.openai.api_server` calls in run_cross_eval.sh /
util/run_benign_eval.sh). It survives `uv pip install --reinstall` of the
instrumentator (it is not part of that package). Re-install after a full venv
re-create with: util/install_vllm_compat.sh.

This is a targeted compat shim for the chosen vLLM 0.11.0 + late-2025 web stack.
The durable alternative is to move vLLM to a release built for starlette 1.x.
"""

from __future__ import annotations


def apply() -> bool:
    """Monkeypatch the instrumentator route walker. Returns True if patched.

    Never raises — a failure here must not break interpreter startup.
    """
    try:
        from prometheus_fastapi_instrumentator import routing as _routing
        from starlette.routing import Match as _Match
    except Exception:
        return False

    if getattr(_routing, "_AAS_PATCHED", False):
        return True

    def _get_route_name(scope, routes, route_name=None):
        for route in routes:
            try:
                match, child_scope = route.matches(scope)
            except Exception:
                continue
            if match == _Match.FULL:
                name = getattr(route, "path", None) or ""
                child_scope = {**scope, **child_scope}
                subroutes = getattr(route, "routes", None)
                if subroutes:
                    child = _get_route_name(child_scope, subroutes, name)
                    name = (name + child) if child is not None else (name or None)
                return name or None
            elif match == _Match.PARTIAL and route_name is None:
                route_name = getattr(route, "path", None)
        return route_name

    _routing._get_route_name = _get_route_name
    _routing._AAS_PATCHED = True
    return True


# Applied on import (the .pth hook does `import aas_vllm_compat`).
apply()
