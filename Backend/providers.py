import math
import logging
import os
import re
from typing import Any

import requests
from fastapi import HTTPException

try:
    from .pin_coordinates import PIN_COORDINATES
except ImportError:
    from pin_coordinates import PIN_COORDINATES

PIN_PATTERN = re.compile(r"^[1-9][0-9]{5}$")
EARTH_RADIUS_KM = 6371.0088
PROVIDER_PAGE_SIZE = 100
logger = logging.getLogger(__name__)


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    value = (
        math.sin(dp / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, value)))


def _supabase_configuration() -> tuple[str, str, str]:
    url = os.getenv("SUPABASE_URL")
    anon_key = os.getenv("SUPABASE_ANON_KEY")
    publishable_key = os.getenv("SUPABASE_PUBLISHABLE_KEY") or anon_key
    if not url or not anon_key or not publishable_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "Provider directory is not configured. Set SUPABASE_URL and "
                "SUPABASE_ANON_KEY in Backend/.env, then restart the backend."
            ),
        )
    if (
        "your-project.supabase.co" in url.lower()
        or "your-supabase" in anon_key.lower()
        or "your-supabase" in publishable_key.lower()
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "Supabase settings still contain example placeholders. Replace "
                "SUPABASE_URL and SUPABASE_ANON_KEY in Backend/.env with your "
                "Supabase project values, then restart the backend."
            ),
        )
    return url.rstrip("/"), publishable_key, anon_key


def _supabase_headers(publishable_key: str, anon_key: str) -> dict[str, str]:
    return {
        "apikey": publishable_key,
        "Authorization": f"Bearer {anon_key}",
    }


def _lookup_pin(pin: str) -> tuple[float, float]:
    try:
        url, publishable_key, anon_key = _supabase_configuration()
    except HTTPException:
        raise

    try:
        response = requests.get(
            f"{url}/rest/v1/postal_codes",
            params={
                "select": "latitude,longitude",
                "pin_code": f"eq.{pin}",
                "limit": 1,
            },
            headers=_supabase_headers(publishable_key, anon_key),
            timeout=10,
        )
        response.raise_for_status()
        rows = response.json()
    except requests.HTTPError as error:
        status_code = error.response.status_code if error.response else "unknown"
        if status_code == 404 and pin in PIN_COORDINATES:
            return PIN_COORDINATES[pin]
        if status_code == 404:
            raise HTTPException(
                status_code=503,
                detail=(
                    "The Supabase postal-code table is missing. Run "
                    "supabase/schema.sql to enable Mumbai-region PIN searches."
                ),
            ) from error
        if status_code in (401, 403):
            raise HTTPException(
                status_code=502,
                detail="Supabase rejected the postal-code lookup. Check the API key and read policy.",
            ) from error
        raise HTTPException(
            status_code=502,
            detail=f"Supabase postal-code lookup failed with HTTP {status_code}.",
        ) from error
    except requests.RequestException as error:
        if pin in PIN_COORDINATES:
            return PIN_COORDINATES[pin]
        logger.exception("Could not connect to Supabase postal-code directory.")
        raise HTTPException(
            status_code=502,
            detail="Could not connect to the Supabase postal-code directory.",
        ) from error

    if not isinstance(rows, list):
        raise HTTPException(
            status_code=502,
            detail="The Supabase postal-code table returned an invalid response.",
        )
    if rows:
        try:
            latitude = float(rows[0]["latitude"])
            longitude = float(rows[0]["longitude"])
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(
                status_code=502,
                detail="The Supabase postal-code record has invalid coordinates.",
            ) from error
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise HTTPException(
                status_code=502,
                detail="The Supabase postal-code record has invalid coordinates.",
            )
        return latitude, longitude

    try:
        return PIN_COORDINATES[pin]
    except KeyError as error:
        raise HTTPException(
            status_code=422,
            detail=(
                "This PIN code is not in the Supabase Mumbai-region postal-code "
                "directory yet."
            ),
        ) from error


def _load_verified_providers() -> list[dict[str, Any]]:
    url = os.getenv("SUPABASE_URL")
    anon_key = os.getenv("SUPABASE_ANON_KEY")
    publishable_key = os.getenv("SUPABASE_PUBLISHABLE_KEY") or anon_key
    if not url or not anon_key or not publishable_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "Provider directory is not configured. Set SUPABASE_URL and "
                "SUPABASE_ANON_KEY in Backend/.env, then restart the backend."
            ),
        )
    if (
        "your-project.supabase.co" in url.lower()
        or "your-supabase" in anon_key.lower()
        or "your-supabase" in publishable_key.lower()
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "Supabase settings still contain example placeholders. Replace "
                "SUPABASE_URL and SUPABASE_ANON_KEY in Backend/.env with your "
                "Supabase project values, then restart the backend."
            ),
        )

    records: list[dict[str, Any]] = []
    offset = 0
    while True:
        try:
            response = requests.get(
                f"{url.rstrip('/')}/rest/v1/providers",
                params={
                    "select": "*",
                    "verified": "eq.true",
                    "order": "name.asc",
                    "limit": PROVIDER_PAGE_SIZE,
                    "offset": offset,
                },
                headers={
                    "apikey": publishable_key,
                    "Authorization": f"Bearer {anon_key}",
                },
                timeout=10,
            )
            response.raise_for_status()
            page = response.json()
        except requests.HTTPError as error:
            status_code = error.response.status_code if error.response else "unknown"
            logger.exception(
                "Supabase provider lookup returned HTTP %s.",
                status_code,
            )
            if status_code in (401, 403):
                detail = (
                    "Supabase rejected the provider lookup. Check the backend "
                    "SUPABASE_URL, SUPABASE_ANON_KEY, and public.providers read policy."
                )
            elif status_code == 404:
                detail = (
                    "Supabase could not find public.providers. Check the table "
                    "and PostgREST schema cache."
                )
            else:
                detail = f"Supabase provider lookup failed with HTTP {status_code}."
            raise HTTPException(status_code=502, detail=detail) from error
        except requests.RequestException as error:
            logger.exception("Could not connect to Supabase provider directory.")
            raise HTTPException(
                status_code=502,
                detail=(
                    "Could not connect to Supabase. Check backend network "
                    "access and SUPABASE_URL."
                ),
            ) from error

        if not isinstance(page, list):
            raise HTTPException(
                status_code=502,
                detail="The provider directory returned an invalid response.",
            )
        records.extend(
            provider
            for provider in page
            if isinstance(provider, dict) and provider.get("verified") is True
        )
        if len(page) < PROVIDER_PAGE_SIZE:
            break
        offset += PROVIDER_PAGE_SIZE

    return records


def find_providers(pin: str, radius_km: float) -> dict[str, Any]:
    if not PIN_PATTERN.fullmatch(pin):
        raise HTTPException(
            status_code=422,
            detail="PIN must be a six-digit Indian PIN code.",
        )
    if not math.isfinite(radius_km) or radius_km <= 0 or radius_km > 100:
        raise HTTPException(
            status_code=422,
            detail="Radius must be greater than 0 and no more than 100 kilometres.",
        )

    latitude, longitude = _lookup_pin(pin)
    providers = []
    for provider in _load_verified_providers():
        try:
            distance = _distance_km(
                latitude,
                longitude,
                float(provider["latitude"]),
                float(provider["longitude"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        if distance <= radius_km:
            providers.append(
                {
                    **provider,
                    "distance_km": round(distance, 1),
                    "distance_type": "straight_line",
                }
            )

    providers.sort(key=lambda provider: provider["distance_km"])
    return {
        "origin": {"pin": pin, "latitude": latitude, "longitude": longitude},
        "radius_km": radius_km,
        "distance_type": "straight_line",
        "providers": providers,
    }
