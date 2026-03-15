import os
import asyncio
import json
import re
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Optional

import httpx
import anthropic
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
REBRICKABLE_API_KEY = os.getenv("REBRICKABLE_API_KEY", "")

if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY environment variable is required")

claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ─── Results store (in-memory, 48h TTL) ──────────────────────────────────────
_results_store: dict[str, dict] = {}

def store_results(data: dict) -> str:
    share_id = uuid.uuid4().hex[:8]
    _results_store[share_id] = {
        "data": data,
        "expires": datetime.utcnow() + timedelta(hours=48),
    }
    # Prune expired entries
    now = datetime.utcnow()
    for k in [k for k, v in _results_store.items() if now > v["expires"]]:
        del _results_store[k]
    return share_id

def get_stored_results(share_id: str) -> Optional[dict]:
    entry = _results_store.get(share_id)
    if not entry:
        return None
    if datetime.utcnow() > entry["expires"]:
        del _results_store[share_id]
        return None
    return entry["data"]

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="LEGO Recommender")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Models ──────────────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    input_type: str   # "manual" | "xml" | "url"
    data: str


# ─── Parsers ─────────────────────────────────────────────────────────────────

def parse_manual(data: str) -> list[str]:
    """Accept comma, newline, or space-separated set numbers."""
    raw = re.split(r"[,\n\r\s]+", data.strip())
    return [s.strip() for s in raw if s.strip()]


def parse_xml(data: str) -> list[str]:
    """Parse BrickLink Wanted List / Collection XML export."""
    sets = []
    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        raise HTTPException(status_code=400, detail=f"Invalid XML: {e}")

    for item in root.iter("ITEM"):
        item_type = (item.findtext("ITEMTYPE") or "").strip().upper()
        if item_type != "S":
            continue
        set_id = (item.findtext("ITEMID") or item.findtext("ITEMNO") or "").strip()
        if set_id and set_id not in sets:
            sets.append(set_id)

    if not sets:
        raise HTTPException(
            status_code=400,
            detail="No sets found in XML. Make sure you exported a Set collection (ITEMTYPE=S).",
        )
    return sets


async def scrape_bricklink_url(url: str) -> list[str]:
    """
    Scrape set numbers from a public BrickLink collection or wishlist URL.
    BrickLink public collection: https://www.bricklink.com/collection.asp?u=USERNAME
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=20.0) as client:
        try:
            resp = await client.get(url)
        except httpx.RequestError as e:
            raise HTTPException(status_code=400, detail=f"Could not reach URL: {e}")

        if resp.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=f"URL returned status {resp.status_code}. Make sure your collection is public.",
            )

        soup = BeautifulSoup(resp.text, "lxml")
        sets = []

        # BrickLink catalog links for sets look like:
        # /v2/catalog/catalogitem.page?S=75192  or  catalogitem.page?S=75192-1
        for a in soup.find_all("a", href=True):
            href = a["href"]
            match = re.search(r"[?&]S=([0-9]+(?:-[0-9]+)?)", href, re.IGNORECASE)
            if match:
                snum = match.group(1)
                if snum not in sets:
                    sets.append(snum)

        if not sets:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No sets found at that URL. "
                    "Ensure the collection is public and the URL points to a set collection page."
                ),
            )
        return sets


# ─── Rebrickable enrichment ───────────────────────────────────────────────────

def normalize_set_num(s: str) -> str:
    return s if "-" in s else f"{s}-1"


async def fetch_set_info(set_num: str, client: httpx.AsyncClient) -> dict:
    """Fetch set metadata from Rebrickable. Returns minimal dict on failure."""
    snum = normalize_set_num(set_num)
    try:
        resp = await client.get(
            f"https://rebrickable.com/api/v3/lego/sets/{snum}/",
            headers={"Authorization": f"key {REBRICKABLE_API_KEY}"},
            timeout=8.0,
        )
        if resp.status_code == 200:
            d = resp.json()
            return {
                "set_num": snum,
                "name": d.get("name", ""),
                "year": d.get("year"),
                "num_parts": d.get("num_parts"),
                "theme_id": d.get("theme_id"),
                "set_img_url": d.get("set_img_url"),
            }
    except Exception:
        pass
    return {"set_num": snum, "name": ""}


async def enrich_sets(set_numbers: list[str]) -> list[dict]:
    """Enrich up to 60 sets with Rebrickable data (batched to respect rate limits)."""
    if not REBRICKABLE_API_KEY:
        return [{"set_num": normalize_set_num(s)} for s in set_numbers]

    # Cap at 60 to avoid hammering the API
    capped = set_numbers[:60]
    async with httpx.AsyncClient() as client:
        # Rebrickable allows ~100 req/min on free tier; batching to be safe
        batch_size = 10
        results = []
        for i in range(0, len(capped), batch_size):
            batch = capped[i : i + batch_size]
            chunk = await asyncio.gather(*[fetch_set_info(s, client) for s in batch])
            results.extend(chunk)
            if i + batch_size < len(capped):
                await asyncio.sleep(0.5)
    return results


# ─── Claude recommendation engine ────────────────────────────────────────────

SYSTEM_PROMPT = """You are a world-class LEGO expert and recommendation engine.
You have deep knowledge of every LEGO theme, subtheme, set, and the collector community.
You always respond with valid JSON only — no markdown, no extra text."""

def build_prompt(sets_data: list[dict], total_count: int, owned_nums: set[str]) -> str:
    collection_lines = []
    for s in sets_data:
        parts = f", {s['num_parts']} parts" if s.get("num_parts") else ""
        year = f" ({s['year']})" if s.get("year") else ""
        name = s.get("name") or ""
        collection_lines.append(f"- {s['set_num']} {name}{year}{parts}")

    collection_str = "\n".join(collection_lines) if collection_lines else "(set numbers only, no metadata)"
    owned_list = ", ".join(sorted(owned_nums)) if owned_nums else "none"

    return f"""The user owns {total_count} LEGO sets. Here is their collection (enriched data for up to 60 sets shown):

{collection_str}

OWNED SET NUMBERS (do NOT recommend any of these): {owned_list}

Analyze their collection and return a JSON object with exactly this structure:

{{
  "collection_insights": {{
    "top_themes": [
      {{"name": "Star Wars", "count": 12}},
      {{"name": "Technic", "count": 8}}
    ],
    "total_estimated_parts": 45000,
    "avg_parts_per_set": 900,
    "year_range": "2015–2024",
    "collector_type": "Flagship Builder"
  }},
  "collection_profile": "2-3 sentence summary of this collector's taste, style, and focus areas",
  "missing_from_collection": [
    {{
      "set_num": "XXXXX-1",
      "name": "Full Official Set Name",
      "theme": "Theme name",
      "year": 2024,
      "reason": "Specific reason referencing sets they already own — what gap this fills or series it completes",
      "price_range": "$XX–$XX",
      "fit_score": 9
    }}
  ],
  "people_like_you": [
    {{
      "set_num": "XXXXX-1",
      "name": "Full Official Set Name",
      "theme": "Theme name",
      "year": 2024,
      "reason": "Why fans with this collector's profile tend to love this set",
      "price_range": "$XX–$XX",
      "fit_score": 8
    }}
  ]
}}

Rules:
- top_themes: list the top 5 themes by set count, in descending order
- collector_type: a short 2-3 word label (e.g. "Flagship Builder", "Theme Completionist", "Technic Enthusiast")
- Provide exactly 5 sets in each recommendation list
- NEVER recommend a set the user already owns — cross-check every set_num against the OWNED SET NUMBERS list above
- missing_from_collection: sets that complete series, fill theme gaps, or are iconic missing pieces given their taste
- people_like_you: sets popular among LEGO fans with a similar collecting style, possibly outside their main themes
- Reasons must be specific and reference actual sets in their collection
- fit_score is 1–10 (10 = perfect fit)
- All set_num values must be real, valid LEGO set numbers
- Return valid JSON only"""


async def get_recommendations(sets_data: list[dict], total_count: int, owned_nums: set[str]) -> dict:
    prompt = build_prompt(sets_data, total_count, owned_nums)

    response = claude.messages.create(
        model="claude-opus-4-6",
        max_tokens=2500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse Claude response: {e}\n\nRaw: {raw[:500]}")


# ─── API endpoints ────────────────────────────────────────────────────────────

@app.post("/api/recommend")
@limiter.limit("5/hour")
async def recommend(request: Request, body: RecommendRequest):
    # 1. Parse input
    if body.input_type == "manual":
        set_numbers = parse_manual(body.data)
    elif body.input_type == "xml":
        set_numbers = parse_xml(body.data)
    elif body.input_type == "url":
        set_numbers = await scrape_bricklink_url(body.data)
    else:
        raise HTTPException(status_code=400, detail="input_type must be 'manual', 'xml', or 'url'")

    if not set_numbers:
        raise HTTPException(status_code=400, detail="No set numbers found in input")

    total_count = len(set_numbers)

    # Build a normalized set of owned numbers for fast lookup
    owned_nums = {normalize_set_num(s) for s in set_numbers}

    # 2. Enrich with Rebrickable (optional)
    sets_data = await enrich_sets(set_numbers)

    # 3. Ask Claude (pass owned set numbers so it won't recommend them)
    recommendations = await get_recommendations(sets_data, total_count, owned_nums)

    # 4. Validate recommended sets against Rebrickable:
    #    - override name with the real official name
    #    - attach image URL
    #    - remove any set Claude hallucinated into the owned collection
    if REBRICKABLE_API_KEY:
        all_rec_sets = (
            recommendations.get("missing_from_collection", []) +
            recommendations.get("people_like_you", [])
        )
        rec_set_nums = [s["set_num"] for s in all_rec_sets if s.get("set_num")]
        async with httpx.AsyncClient() as client:
            rb_results = await asyncio.gather(
                *[fetch_set_info(s, client) for s in rec_set_nums],
                return_exceptions=True
            )
        # Map normalized set_num → Rebrickable data
        rb_map = {
            r["set_num"]: r
            for r in rb_results
            if isinstance(r, dict) and r.get("set_num")
        }
        for section in ("missing_from_collection", "people_like_you"):
            validated = []
            for item in recommendations.get(section, []):
                snum = normalize_set_num(item.get("set_num", ""))
                # Drop sets the user already owns (safety net)
                if snum in owned_nums:
                    continue
                rb = rb_map.get(snum)
                if rb:
                    # Override with authoritative Rebrickable data
                    if rb.get("name"):
                        item["name"] = rb["name"]
                    if rb.get("set_img_url"):
                        item["img_url"] = rb["set_img_url"]
                validated.append(item)
            recommendations[section] = validated

    result = {
        "set_count": total_count,
        "rebrickable_enriched": bool(REBRICKABLE_API_KEY),
        "recommendations": recommendations,
    }

    result["share_id"] = store_results(result)
    return result


@app.get("/api/results/{share_id}")
async def shared_results(share_id: str):
    data = get_stored_results(share_id)
    if not data:
        raise HTTPException(status_code=404, detail="Results not found or expired (48h limit).")
    return data


@app.get("/health")
def health():
    return {"status": "ok", "rebrickable": bool(REBRICKABLE_API_KEY)}


# Serve frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")
