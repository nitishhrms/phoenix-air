import os
import concurrent.futures
import httpx
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv(
    "EXTERNAL_API_BASE",
    "https://zz1mpoguje.execute-api.us-east-1.amazonaws.com/default/airline-assessment",
)


def search_flights(src: str, dst: str, date: str) -> dict:
    """
    GET flights for a route and date.
    Returns { "flights": [...] } on success.
    Returns { "error": "...", "code": "NO_FLIGHTS"|"INVALID_DATE"|"API_ERROR" } on failure.
    """
    try:
        resp = httpx.get(
            BASE_URL,
            params={"src": src, "dst": dst, "date": date},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return {"flights": data.get("flights", [])}
        if resp.status_code == 404:
            return {"error": "No flights available on this route.", "code": "NO_FLIGHTS"}
        if resp.status_code == 400:
            return {"error": "Invalid or past date.", "code": "INVALID_DATE"}
        return {"error": f"API returned {resp.status_code}.", "code": "API_ERROR"}
    except httpx.RequestError as e:
        return {"error": f"Could not reach flight service: {e}", "code": "API_ERROR"}


def search_flights_multi(src: str, dst: str, dates: list) -> dict:
    """
    Search flights for multiple dates in parallel.
    Returns {"flights_by_date": {"YYYY-MM-DD": [...], ...}}
    """
    def fetch(date):
        result = search_flights(src, dst, date)
        return date, result.get("flights", [])

    flights_by_date = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(dates)) as executor:
        futures = {executor.submit(fetch, d): d for d in dates}
        for future in concurrent.futures.as_completed(futures):
            try:
                date, flights = future.result()
                if flights:
                    flights_by_date[date] = flights
            except Exception:
                pass

    return {"flights_by_date": flights_by_date}


def book_flight(src: str, dst: str, date: str,
                flight_id: str, first_name: str, last_name: str) -> dict:
    """
    POST a booking.
    Returns the raw API response dict, or { "error": "...", "code": "..." }.
    """
    try:
        resp = httpx.post(
            BASE_URL,
            params={"src": src, "dst": dst, "date": date},
            json={
                "flightId": flight_id,
                "passenger": {"firstName": first_name, "lastName": last_name},
                "date": date,
            },
            timeout=10,
        )
        if resp.status_code in (200, 201):
            return resp.json()
        return {"error": f"Booking failed with status {resp.status_code}.", "code": "BOOKING_ERROR"}
    except httpx.RequestError as e:
        return {"error": f"Could not reach booking service: {e}", "code": "API_ERROR"}
