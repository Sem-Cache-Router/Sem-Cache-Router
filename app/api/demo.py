# The showcase page.
#
# GET / returns app/static/index.html, one self contained page that shows the
# whole gateway working in a browser: ask questions and see miss, exact hit or
# semantic hit with the time taken, the live counters from /metrics, the
# dependency states from /health, failover against the mock vendor, and the
# cache flush. It is served by the gateway itself, so the page and the API share
# an origin and nothing extra has to run.
#
# The page reads only what the API already exposes and holds no secret. The
# admin key it can send is typed in by the person using it and never stored.

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

PAGE = Path(__file__).resolve().parent.parent / "static" / "index.html"

router = APIRouter(tags=["demo"])


@router.get("/", include_in_schema=False)
async def demo_page() -> FileResponse:
    """Serve the showcase page, telling browsers to check for a newer one every time.

    Without it a browser may keep showing an old copy after a deployment, as
    one did after the redesign; with no-cache it revalidates, and the file's
    ETag keeps that cheap when nothing changed.
    """
    return FileResponse(PAGE, media_type="text/html", headers={"Cache-Control": "no-cache"})
