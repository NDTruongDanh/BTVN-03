"""ReAct agent: Reason -> Act -> Observe -> Reason again.

Each loop the policy (standing in for the LLM) picks ONE next tool from the
live observation. The harness validates permission BEFORE execution and
checks completion / loop / stall / budget AFTER each observation.

Also provides build_react_langchain_agent() which wires the same tools and
guards into LangChain's create_agent for use with a real model
(MODEL_NAME env var, e.g. "openai:gpt-4o-mini").
"""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass
from typing import Any, Dict, List, Optional

from flight_tools import (
    TOOLS,
    raw_book_seat,
    raw_check_seat,
    raw_get_booking,
    raw_pay,
    raw_search_flights,
    reset_database,
)
from harness import (
    AgentResult,
    BookingConstraints,
    BudgetTracker,
    LoopDetector,
    PermissionChecker,
    TraceLogger,
    build_handoff,
    check_completion,
    cross_check_price,
)

SYSTEM_PROMPT = (
    "Use tools for every fact, including destination names. "
    "Never invent flight IDs, prices, or booking codes. "
    "After pay, always call get_booking to verify."
)


def _as_flight_dict(obj: Any) -> Optional[Dict[str, Any]]:
    """Narrow an untyped JSON value to a flight dict for type-checkers."""
    return obj if isinstance(obj, dict) else None


def _as_str(obj: Any) -> Optional[str]:
    return obj if isinstance(obj, str) else None


def _pick_next(state: Dict[str, Any],
               constraints: BookingConstraints) -> Optional[Dict[str, Any]]:
    """Rule-based stand-in for the model: decide the next single tool call.

    Only flights that already satisfy the static constraints in the search
    listing are verified with check_seat (listing already carries price and
    time, so re-checking a flight that visibly violates budget/time is
    wasted work). Timeouts are retried on the SAME flight (per the tool
    hint) so the harness loop detector can fire; sold_out moves on.
    """
    if not state.get("searched"):
        return {"tool": "search_flights",
                "args": {"origin": constraints.origin,
                         "destination": constraints.destination,
                         "date": constraints.date}}
    if state.get("held_code"):
        if not state.get("paid"):
            return {"tool": "pay", "args": {"code": state["held_code"]}}
        return {"tool": "get_booking", "args": {"code": state["held_code"]}}
    if state.get("retry"):
        flight_id = state.pop("retry")
        return {"tool": "check_seat", "args": {"flight_id": flight_id}}
    # Verify cheapest satisfying candidates first.
    for cand in state.get("candidates", []):
        if cand["flight_id"] not in state.get("checked", {}) \
                and cand["flight_id"] not in state.get("failed", set()) \
                and constraints.is_ok(cand):
            return {"tool": "check_seat", "args": {"flight_id": cand["flight_id"]}}
    if state.get("best") and not state.get("held_code"):
        return {"tool": "book_seat",
                "args": {"flight_id": state["best"]["flight_id"]}}
    return None  # model stops calling tools


def run_react_task(constraints: BookingConstraints | None = None,
                   scenario: str = "standard",
                   max_steps: int = 12,
                   auto_approve: bool = False,
                   verbose: bool = True) -> AgentResult:
    constraints = constraints or BookingConstraints()
    reset_database(scenario)
    trace = TraceLogger()
    loop = LoopDetector()
    budget = BudgetTracker(max_steps=max_steps)
    gate = PermissionChecker(auto_approve=auto_approve)
    tried: List[str] = []
    side_effects: List[str] = []

    state: Dict[str, Any] = {"searched": False, "candidates": [],
                             "checked": {}, "failed": set(),
                             "best": None, "held_code": None,
                             "paid": False, "seen_price": None}
    flight_by_id: Dict[str, Dict[str, Any]] = {}

    for round_no in range(1, max_steps + 1):
        budget.tick()
        action = _pick_next(state, constraints)
        if action is None:  # model stopped -> verify with CODE, never trust words
            done, msg = check_completion(state.get("held_code"), constraints)
            if done:
                return AgentResult("SUCCESS", state["held_code"],
                                   f"ReAct done, {msg}", budget.steps, trace)
            handoff = build_handoff("FAILED", constraints, trace, tried,
                                    "No further action proposed but goal unverified. "
                                    "Which flight should be booked?",
                                    side_effects)
            return AgentResult("FAILED", state.get("held_code"),
                               f"ReAct stopped early: {msg}", budget.steps,
                               trace, handoff)

        tool_raw: Any = action["tool"]
        args_raw: Any = action["args"]
        tool: str = tool_raw if isinstance(tool_raw, str) else str(tool_raw)
        args: Dict[str, Any] = dict(args_raw) if isinstance(args_raw, dict) else {}
        # ---- checklist #0: permission BEFORE execution ----
        # Narrow to Optional[Dict[str, Any]] so PermissionChecker.check
        # never receives a stray `str` (was: Dict | Any | str | None).
        flight_ctx: Optional[Dict[str, Any]] = None
        fid_raw: Any = args.get("flight_id")
        if isinstance(fid_raw, str):
            flight_ctx = _as_flight_dict(flight_by_id.get(fid_raw))
        if flight_ctx is None and tool == "book_seat":
            flight_ctx = _as_flight_dict(state.get("best"))
        allowed, why = gate.check(tool, args, flight_ctx)
        if not allowed:
            handoff = build_handoff("NEEDS_APPROVAL", constraints, trace, tried,
                                    f"Approve {tool}({args})? {why}",
                                    side_effects)
            return AgentResult("NEEDS_APPROVAL", state.get("held_code"),
                               f"Waiting approval: {why}", budget.steps,
                               trace, handoff)

        # ---- execute ----
        obs: Dict[str, Any]
        if tool == "search_flights":
            obs = raw_search_flights(
                origin=str(args.get("origin", "")),
                destination=str(args.get("destination", "")),
                date=str(args.get("date", "")),
            )
        elif tool == "check_seat":
            obs = raw_check_seat(flight_id=str(args.get("flight_id", "")))
        elif tool == "book_seat":
            obs = raw_book_seat(flight_id=str(args.get("flight_id", "")))
        elif tool == "pay":
            obs = raw_pay(code=str(args.get("code", "")))
        elif tool == "get_booking":
            obs = raw_get_booking(code=str(args.get("code", "")))
        else:
            obs = {"status": "invalid_param", "hint": f"unknown tool {tool}"}
        trace.log(round_no, tool, args, obs)
        tried.append(f"{tool}({args}) -> {obs.get('status')}")
        if verbose:
            print(f"[V{round_no}] {tool}({args}) -> {obs}")

        # ---- fold observation into state ----
        progress: Any = None
        if tool == "search_flights" and obs.get("status") == "ok":
            state["searched"] = True
            flights_raw: Any = obs.get("flights", [])
            flights: List[Dict[str, Any]] = (
                [f for f in flights_raw if isinstance(f, dict)]
                if isinstance(flights_raw, list) else []
            )
            # rank: satisfying constraints first, then cheapest
            flights.sort(
                key=lambda f: (0 if constraints.is_ok(f) else 1,
                               int(f.get("price", 0)) if isinstance(f.get("price"), (int, float)) else 0)
            )
            state["candidates"] = flights
            flight_by_id = {
                str(f["flight_id"]): f for f in flights
                if isinstance(f.get("flight_id"), str)
            }
            progress = constraints.progress_score(
                next((f for f in flights if constraints.is_ok(f)), None))
        elif tool == "check_seat":
            if obs.get("status") == "ok":
                f_raw: Any = obs.get("flight")
                f: Dict[str, Any] = f_raw if isinstance(f_raw, dict) else {}
                fid = f.get("flight_id")
                if isinstance(fid, str):
                    checked_map: Any = state.get("checked")
                    if isinstance(checked_map, dict):
                        checked_map[fid] = f
                    flight_by_id[fid] = f
                best_raw: Any = state.get("best")
                best_dict = _as_flight_dict(best_raw)
                f_price: Any = f.get("price")
                best_price: Any = best_dict.get("price") if best_dict else None
                if constraints.is_ok(f) and (
                        best_dict is None
                        or (isinstance(f_price, (int, float))
                            and isinstance(best_price, (int, float))
                            and f_price < best_price)):
                    state["best"] = f
                progress = constraints.progress_score(state.get("best"))
            elif obs.get("status") == "error":
                # Transient failure: retry SAME action so LoopDetector fires
                # if the model keeps retrying without progress (slide V2-V4).
                retry_raw: Any = args.get("flight_id")
                if isinstance(retry_raw, str):
                    state["retry"] = retry_raw
                progress = constraints.progress_score(state.get("best"))
            else:
                fail_raw: Any = args.get("flight_id")
                failed_set: Any = state.get("failed")
                if isinstance(fail_raw, str) and isinstance(failed_set, set):
                    failed_set.add(fail_raw)
                progress = constraints.progress_score(state.get("best"))
        elif tool == "book_seat":
            if obs.get("status") == "held":
                code_raw: Any = obs.get("code")
                booking_raw: Any = obs.get("booking")
                booking_dict: Dict[str, Any] = booking_raw if isinstance(booking_raw, dict) else {}
                if isinstance(code_raw, str):
                    state["held_code"] = code_raw
                    side_effects.append(f"held {code_raw}")
                price_raw: Any = booking_dict.get("price")
                if isinstance(price_raw, int):
                    state["seen_price"] = price_raw
                else:
                    try:
                        state["seen_price"] = int(price_raw) if price_raw is not None else None
                    except (TypeError, ValueError):
                        state["seen_price"] = None
            else:
                best_now = _as_flight_dict(state.get("best"))
                if best_now is not None:
                    best_fid = best_now.get("flight_id")
                    failed_set2: Any = state.get("failed")
                    if isinstance(best_fid, str) and isinstance(failed_set2, set):
                        failed_set2.add(best_fid)
                    state["best"] = None
            progress = 4 if state.get("held_code") else constraints.progress_score(state.get("best"))
        elif tool == "pay":
            if obs.get("status") == "paid":
                state["paid"] = True
                code_for_pay: Any = args.get("code")
                if isinstance(code_for_pay, str):
                    side_effects.append(f"paid {code_for_pay}")
            progress = 5 if state.get("paid") else 4
        elif tool == "get_booking":
            progress = 6

        # ---- checklist #1: completion (computational sensor) ----
        held_for_check: Any = state.get("held_code")
        held_code_opt: Optional[str] = held_for_check if isinstance(held_for_check, str) else None
        done, msg = check_completion(held_code_opt, constraints)
        if done:
            seen_raw: Any = state.get("seen_price")
            seen_price_opt: Optional[int] = seen_raw if isinstance(seen_raw, int) else None
            ok, cmsg = cross_check_price(held_code_opt, seen_price_opt)
            if verbose:
                print(f"Verified: {msg} | cross-check: {cmsg}")
            if ok:
                return AgentResult("SUCCESS", held_code_opt,
                                   f"ReAct done, {msg}", budget.steps, trace)
            handoff = build_handoff("FAILED", constraints, trace, tried,
                                    f"Price mismatch on {held_code_opt}: {cmsg}. Proceed?",
                                    side_effects)
            return AgentResult("FAILED", held_code_opt, cmsg,
                               budget.steps, trace, handoff)

        # ---- checklist #2/#3: loop / stall (compare tool+args, not obs) ----
        flag = loop.check(tool, args, progress)
        if flag == "LOOP":
            handoff = build_handoff("LOOP", constraints, trace, tried,
                                    f"Action {tool}({args}) repeated with no progress. "
                                    "Change flight or escalate?", side_effects)
            return AgentResult("LOOP", state.get("held_code"),
                               f"Loop detected at V{round_no}", budget.steps,
                               trace, handoff)
        if flag == "STALL":
            handoff = build_handoff("STALLED", constraints, trace, tried,
                                    "Progress metric frozen for 5 rounds. New direction?",
                                    side_effects)
            return AgentResult("STALLED", state.get("held_code"),
                               f"Stall detected at V{round_no}", budget.steps,
                               trace, handoff)

        # ---- checklist #4: budget LAST ----
        over, reason = budget.exceeded()
        if over:
            handoff = build_handoff("BUDGET_EXCEEDED", constraints, trace, tried,
                                    f"Budget hit ({reason}). Extend budget or stop?",
                                    side_effects)
            return AgentResult("BUDGET_EXCEEDED", state.get("held_code"),
                               reason, budget.steps, trace, handoff)

    handoff = build_handoff("BUDGET_EXCEEDED", constraints, trace, tried,
                            "Step budget exhausted. Extend or stop?", side_effects)
    return AgentResult("BUDGET_EXCEEDED", state.get("held_code"),
                       "max steps reached", budget.steps, trace, handoff)


# ---------------------------------------------------------------------------
# LangChain wiring (requires MODEL_NAME, e.g. "openai:gpt-4o-mini")
# ---------------------------------------------------------------------------

def build_react_langchain_agent():
    """Create a real ReAct agent with LangChain. Needs MODEL_NAME set."""
    from langchain.agents import create_agent
    from langchain.agents.middleware import ModelCallLimitMiddleware

    model = os.environ["MODEL_NAME"]  # raises if missing -> intentional
    return create_agent(model=model, tools=TOOLS, system_prompt=SYSTEM_PROMPT,
                        middleware=[ModelCallLimitMiddleware(run_limit=12,
                                                             exit_behavior="end")])


if __name__ == "__main__":
    r = run_react_task(scenario="standard", auto_approve=True)
    print(f"\nResult: {r.status} | {r.message} | steps={r.steps}")
