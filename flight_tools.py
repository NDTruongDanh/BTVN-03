"""Mock flight-booking tools with structured JSON observations.

All tools return plain dicts with an explicit ``status`` field so the agent
never has to guess what an empty or ambiguous result means:

- ``ok``            : real data is present
- ``invalid_param`` : caller sent a bad argument (with a ``hint`` to recover)
- ``not_found``     : referenced flight / booking does not exist
- ``sold_out``      : flight exists but has no seat left
- ``error``         : simulated infrastructure failure (e.g. timeout)

This file exposes both RAW python functions (used by the deterministic
offline policies) and LangChain ``@tool`` wrappers (used by real LLM
agents built with ``create_agent``).
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List

from langchain_core.tools import tool

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# ---------------------------------------------------------------------------
# In-memory database
# ---------------------------------------------------------------------------

_BASE_FLIGHTS: List[Dict[str, Any]] = [
    {
        "flight_id": "VN122",
        "origin": "SGN",
        "destination": "DAD",
        "date": "2026-10-07",
        "depart_time": "08:10",
        "price": 1_850_000,
        "seats_left": 3,
        "refundable": True,
    },
    {
        "flight_id": "VJ604",
        "origin": "SGN",
        "destination": "DAD",
        "date": "2026-10-07",
        "depart_time": "09:30",
        "price": 2_080_000,
        "seats_left": 5,
        "refundable": True,
    },
    {
        "flight_id": "QH118",
        "origin": "SGN",
        "destination": "DAD",
        "date": "2026-10-07",
        "depart_time": "15:40",
        "price": 1_640_000,
        "seats_left": 4,
        "refundable": True,
    },
    {
        "flight_id": "VN134",
        "origin": "SGN",
        "destination": "DAD",
        "date": "2026-10-07",
        "depart_time": "10:20",
        "price": 1_950_000,
        "seats_left": 1,
        "refundable": False,  # non-refundable -> triggers approval gate
    },
]

FLIGHTS: Dict[str, Dict[str, Any]] = {}
BOOKINGS: Dict[str, Dict[str, Any]] = {}
_BOOKING_SEQ = 0

# Fault-injection flags used by evaluation scenarios.
FAULTS: Dict[str, Any] = {
    "check_seat_timeout": False,  # every check_seat returns timeout error
    "sell_out_after_search": False,  # VN122 sells out right after first search
    "searched_once": False,
}


def reset_database(scenario: str = "standard") -> None:
    """Reset mock DB. Scenario tunes faults and inventory."""
    global FLIGHTS, BOOKINGS, _BOOKING_SEQ, FAULTS
    FLIGHTS = {f["flight_id"]: copy.deepcopy(f) for f in _BASE_FLIGHTS}
    BOOKINGS = {}
    _BOOKING_SEQ = 0
    FAULTS = {
        "check_seat_timeout": False,
        "sell_out_after_search": False,
        "searched_once": False,
    }
    if scenario == "dynamic_sold_out":
        # VN122 looks available in search but is gone when we act.
        FAULTS["sell_out_after_search"] = True
    elif scenario == "over_budget":
        # Push every morning flight above the 2M budget.
        for f in FLIGHTS.values():
            if f["depart_time"] < "12:00":
                f["price"] = 2_300_000
    elif scenario == "timeout_loop":
        FAULTS["check_seat_timeout"] = True
    elif scenario == "approval_only":
        # Only the non-refundable option remains bookable in the morning.
        for fid in ("VN122", "VJ604"):
            FLIGHTS[fid]["seats_left"] = 0
    # "standard" keeps the base inventory untouched.


def _next_code() -> str:
    global _BOOKING_SEQ
    _BOOKING_SEQ += 1
    return f"BKG{_BOOKING_SEQ:03d}"


# ---------------------------------------------------------------------------
# RAW functions (deterministic, testable, no LLM needed)
# ---------------------------------------------------------------------------


def raw_search_flights(origin: str, destination: str, date: str) -> Dict[str, Any]:
    if not DATE_RE.match(date or ""):
        return {
            "status": "invalid_param",
            "param": "date",
            "hint": "Use YYYY-MM-DD, e.g. 2026-10-07",
        }
    if FAULTS["sell_out_after_search"] and not FAULTS["searched_once"]:
        FAULTS["searched_once"] = True
        # First search sees VN122, then it immediately sells out.
        flights = [
            f for f in FLIGHTS.values()
            if f["origin"] == origin and f["destination"] == destination and f["date"] == date
        ]
        FLIGHTS["VN122"]["seats_left"] = 0
        return {"status": "ok", "flights": copy.deepcopy(flights)}
    flights = [
        f for f in FLIGHTS.values()
        if f["origin"] == origin and f["destination"] == destination and f["date"] == date
    ]
    return {"status": "ok", "flights": copy.deepcopy(flights)}


def raw_check_seat(flight_id: str) -> Dict[str, Any]:
    if FAULTS["check_seat_timeout"]:
        return {"status": "error", "error": "timeout",
                "hint": "Seat service timed out, retry or pick another flight"}
    flight = FLIGHTS.get(flight_id)
    if flight is None:
        return {"status": "not_found", "flight_id": flight_id,
                "hint": "Unknown flight_id, use one from search_flights"}
    if flight["seats_left"] <= 0:
        return {"status": "sold_out", "flight_id": flight_id,
                "hint": "No seat left, pick another flight"}
    return {"status": "ok", "flight": copy.deepcopy(flight)}


def raw_book_seat(flight_id: str) -> Dict[str, Any]:
    flight = FLIGHTS.get(flight_id)
    if flight is None:
        return {"status": "not_found", "flight_id": flight_id}
    if flight["seats_left"] <= 0:
        return {"status": "sold_out", "flight_id": flight_id,
                "hint": "Seat just sold out, replan with another flight"}
    flight["seats_left"] -= 1
    code = _next_code()
    BOOKINGS[code] = {
        "code": code,
        "flight_id": flight_id,
        "origin": flight["origin"],
        "destination": flight["destination"],
        "date": flight["date"],
        "depart_time": flight["depart_time"],
        "price": flight["price"],
        "refundable": flight["refundable"],
        "status": "held",
        "paid": False,
    }
    return {"status": "held", "code": code,
            "booking": copy.deepcopy(BOOKINGS[code])}


def raw_pay(code: str) -> Dict[str, Any]:
    booking = BOOKINGS.get(code)
    if booking is None:
        return {"status": "not_found", "code": code}
    booking["paid"] = True
    booking["status"] = "confirmed"
    return {"status": "paid", "code": code,
            "booking": copy.deepcopy(booking)}


def raw_get_booking(code: str) -> Dict[str, Any]:
    booking = BOOKINGS.get(code)
    if booking is None:
        return {"status": "not_found", "code": code}
    return {"status": "ok", "booking": copy.deepcopy(booking)}


# ---------------------------------------------------------------------------
# LangChain tool wrappers (for real LLM agents via create_agent)
# ---------------------------------------------------------------------------


@tool
def search_flights(origin: str, destination: str, date: str) -> dict:
    """Search available flights. Date must be YYYY-MM-DD."""
    return raw_search_flights(origin, destination, date)


@tool
def check_seat(flight_id: str) -> dict:
    """Check seat availability and price for a flight_id from search results."""
    return raw_check_seat(flight_id)


@tool
def book_seat(flight_id: str) -> dict:
    """Hold one seat on a flight. Returns a booking code with status 'held'."""
    return raw_book_seat(flight_id)


@tool
def pay(code: str) -> dict:
    """Pay for a held booking code. Booking becomes 'confirmed' and paid."""
    return raw_pay(code)


@tool
def get_booking(code: str) -> dict:
    """Read back a booking by code to verify status, payment and price."""
    return raw_get_booking(code)


TOOLS = [search_flights, check_seat, book_seat, pay, get_booking]
