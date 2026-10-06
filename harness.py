"""Shared harness layers for the flight-booking agent.

Checklist order after each observation (budget checked LAST so that every
failure is not misreported as budget-exceeded):
  0. before tool execution : permission  -> NEEDS_APPROVAL (normal)
  1. after observation     : completion  -> SUCCESS (normal)
  2. after observation     : (tool,args) repeat -> LOOP (abnormal)
  3. after observation     : progress stall     -> STALLED (abnormal)
  4. after observation     : steps/time budget  -> BUDGET_EXCEEDED (abnormal)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time

from flight_tools import raw_get_booking


# ---------------------------------------------------------------------------
# 1. Constraints as data (NOT buried in prompts)
# ---------------------------------------------------------------------------

@dataclass
class BookingConstraints:
    origin: str = "SGN"
    destination: str = "DAD"
    date: str = "2026-10-07"
    depart_before: str = "12:00"  # morning flights only
    max_price: int = 2_000_000

    def satisfies(self, flight: Any) -> Tuple[bool, List[str]]:
        reasons: List[str] = []
        if not isinstance(flight, dict):
            return False, [f"invalid flight record {flight!r}"]
        if flight.get("origin") != self.origin:
            reasons.append(f"wrong origin {flight.get('origin')}")
        if flight.get("destination") != self.destination:
            reasons.append(f"wrong destination {flight.get('destination')}")
        if flight.get("date") != self.date:
            reasons.append(f"wrong date {flight.get('date')}")
        if str(flight.get("depart_time", "")) >= self.depart_before:
            reasons.append(f"departs {flight.get('depart_time')} not before {self.depart_before}")
        try:
            price = int(flight.get("price", 10**18))
        except (TypeError, ValueError):
            return False, [f"invalid price {flight.get('price')!r}"]
        if price > self.max_price:
            reasons.append(f"price {flight.get('price')} over budget {self.max_price}")
        return (len(reasons) == 0, reasons)

    def is_ok(self, flight: Any) -> bool:
        ok, _ = self.satisfies(flight)
        return ok

    def progress_score(self, flight_or_none: Any) -> int:
        """Progress metric for stall detection: how many constraints hold."""
        if not isinstance(flight_or_none, dict):
            return 0
        f = flight_or_none
        score = 0
        if f.get("date") == self.date:
            score += 1
        if str(f.get("depart_time", "")) < self.depart_before:
            score += 1
        try:
            price = int(f.get("price", 10**18))
        except (TypeError, ValueError):
            return score
        if price <= self.max_price:
            score += 1
        return score  # 0..3


# ---------------------------------------------------------------------------
# 2. Completion criterion checked by CODE (computational sensor, no tokens)
# ---------------------------------------------------------------------------

def check_completion(code: Optional[str],
                     constraints: BookingConstraints) -> Tuple[bool, str]:
    """Objective rule: booking is confirmed, paid, and satisfies constraints."""
    if not code:
        return False, "no booking code yet"
    obs = raw_get_booking(code)
    if obs.get("status") != "ok":
        return False, f"booking {code} not found"
    b = obs["booking"]
    if b.get("status") != "confirmed" or not b.get("paid"):
        return False, f"booking {code} not confirmed+paid ({b.get('status')}, paid={b.get('paid')})"
    ok, reasons = constraints.satisfies({
        "origin": b["origin"], "destination": b["destination"],
        "date": b["date"], "depart_time": b["depart_time"],
        "price": b["price"],
    })
    if not ok:
        return False, f"booking {code} violates: {'; '.join(reasons)}"
    return True, f"booking {code} confirmed, paid, {b['price']} VND"


def cross_check_price(code: Optional[str], seen_price: Optional[int]) -> Tuple[bool, str]:
    """Anti-hallucination: get_booking price must match the check_seat price."""
    if seen_price is None or not code:
        return True, "nothing to cross-check"
    obs = raw_get_booking(code)
    if obs.get("status") != "ok":
        return False, "cannot cross-check, booking missing"
    actual = obs["booking"]["price"]
    if actual != seen_price:
        return False, f"price mismatch: saw {seen_price}, booking has {actual}"
    return True, "price consistent"


# ---------------------------------------------------------------------------
# 3. Permission check BEFORE execution (checklist #0)
# ---------------------------------------------------------------------------

@dataclass
class PermissionChecker:
    """Gate risky actions. Runs BEFORE the tool executes."""
    approval_limit: int = 2_000_000
    auto_approve: bool = False  # offline eval: simulate a human approver

    def check(self, tool: Any, args: Any,
              flight: Any = None) -> Tuple[bool, str]:
        if tool in ("book_seat", "pay"):
            flight_dict: Dict[str, Any] = flight if isinstance(flight, dict) else {}
            price_any: Any = flight_dict.get("price")
            refundable: Any = flight_dict.get("refundable", True)
            if refundable is False:
                reason = (f"{tool} on non-refundable fare "
                          f"({args}, price={price_any}) requires human approval")
                return (True, "auto-approved (simulated)") if self.auto_approve else (False, reason)
            if isinstance(price_any, (int, float)) and price_any > self.approval_limit:
                reason = (f"{tool} price {price_any} exceeds approval limit "
                          f"{self.approval_limit}")
                return (True, "auto-approved (simulated)") if self.auto_approve else (False, reason)
        return True, "allowed"


# ---------------------------------------------------------------------------
# 4. Loop / stall detector (slide code, fixed comparison bug)
# ---------------------------------------------------------------------------

class LoopDetector:
    def __init__(self, window: int = 6, repeat_k: int = 2, stall_n: int = 5):
        self.recent: deque[tuple[str, str]] = deque(maxlen=window)  # only compare recent window
        self.k, self.n = repeat_k, stall_n
        self.last: Any = None
        self.stall = 0

    def check(self, tool: Any, args: Any, progress: Any) -> Optional[str]:
        args_dict: Dict[str, Any] = args if isinstance(args, dict) else {}
        fp = (str(tool), repr(sorted(args_dict.items())))
        if self.recent.count(fp) + 1 >= self.k:
            return "LOOP"
        self.recent.append(fp)
        self.stall = self.stall + 1 if progress == self.last else 0
        self.last = progress
        if self.stall >= self.n:
            return "STALL"
        return None


# ---------------------------------------------------------------------------
# 5. Budget (checked LAST)
# ---------------------------------------------------------------------------

@dataclass
class BudgetTracker:
    max_steps: int = 12
    max_seconds: float = 60.0
    steps: int = 0
    _start: float = field(default_factory=time.monotonic, init=False)

    def tick(self) -> None:
        self.steps += 1

    def exceeded(self) -> Tuple[bool, str]:
        if self.steps >= self.max_steps:
            return True, f"step budget {self.max_steps} reached"
        if time.monotonic() - self._start > self.max_seconds:
            return True, f"time budget {self.max_seconds}s reached"
        return False, ""


# ---------------------------------------------------------------------------
# 6. Trace + handoff + result
# ---------------------------------------------------------------------------

@dataclass
class TraceLogger:
    entries: List[Dict[str, Any]] = field(default_factory=list)

    def log(self, round_no: int, tool: Any, args: Any,
            observation: Any) -> None:
        self.entries.append({"round": round_no, "tool": tool,
                             "args": args, "observation": observation})

    def dump(self) -> str:
        lines = []
        for e in self.entries:
            lines.append(f"[V{e['round']}] {e['tool']}({e['args']}) -> {e['observation']}")
        return "\n".join(lines)


def build_handoff(status: str, constraints: BookingConstraints,
                  trace: TraceLogger, tried: List[str],
                  question: str,
                  side_effects: List[str]) -> Dict[str, Any]:
    """Good handoff answers in 30 seconds: state, what was tried, one question."""
    return {
        "status": status,
        "state": f"constraints={constraints}, trace_rounds={len(trace.entries)}, "
                 f"side_effects={side_effects or 'none'}",
        "tried": tried,
        "question": question,
        "trace": trace.dump(),
    }


@dataclass
class AgentResult:
    status: str  # SUCCESS | NEEDS_APPROVAL | LOOP | STALLED | BUDGET_EXCEEDED | FAILED
    booking_code: Optional[str]
    message: str
    steps: int
    trace: TraceLogger = field(default_factory=TraceLogger)
    handoff: Optional[Dict[str, Any]] = None
