"""Hybrid agent (ReAct + Plan): plan, execute k steps, replan on new observations.

Mirrors LangChain 1.x TodoListMiddleware behaviour: an explicit todo list is
kept, k steps are executed, then the plan is revised if the observation
changed significantly (sold out / new price / new candidates).
"""

from __future__ import annotations

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

EXECUTORS = {
    "search_flights": raw_search_flights,
    "check_seat": raw_check_seat,
    "book_seat": raw_book_seat,
    "pay": raw_pay,
    "get_booking": raw_get_booking,
}


def _as_flight_dict(obj: Any) -> Optional[Dict[str, Any]]:
    """Narrow an untyped JSON value to a flight dict for type-checkers."""
    return obj if isinstance(obj, dict) else None


def _initial_plan(constraints: BookingConstraints) -> List[Dict[str, Any]]:
    return [
        {"tool": "search_flights",
         "args": {"origin": constraints.origin,
                  "destination": constraints.destination,
                  "date": constraints.date},
         "label": "survey candidates"},
        {"tool": "check_seat", "args": {"flight_id": "VN122"},
         "label": "verify best guess"},
        {"tool": "book_seat", "args": {"flight_id": "<best>"},
         "label": "hold seat"},
        {"tool": "pay", "args": {"code": "<from_book>"}, "label": "pay"},
        {"tool": "get_booking", "args": {"code": "<from_book>"},
         "label": "verify"},
    ]


def _replan(failed: set[str], best_id: Optional[str],
            constraints: BookingConstraints) -> List[Dict[str, Any]]:
    """Build a fresh plan excluding failed flights, preferring current best."""
    target = best_id or "VN134"
    if target in failed:
        target = "VN134" if "VN134" not in failed else "VJ604"
    return [
        {"tool": "check_seat", "args": {"flight_id": target},
         "label": f"re-verify {target}"},
        {"tool": "book_seat", "args": {"flight_id": target},
         "label": "hold seat"},
        {"tool": "pay", "args": {"code": "<from_book>"}, "label": "pay"},
        {"tool": "get_booking", "args": {"code": "<from_book>"},
         "label": "verify"},
    ]


def run_hybrid_task(constraints: BookingConstraints | None = None,
                    scenario: str = "standard",
                    max_steps: int = 14,
                    k: int = 2,
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

    checked: Dict[str, Dict[str, Any]] = {}
    failed: set[str] = set()
    best: Optional[Dict[str, Any]] = None
    held_code: Optional[str] = None
    seen_price: Optional[int] = None
    paid = False
    replans = 0

    plan = _initial_plan(constraints)
    idx, since_replan = 0, 0
    round_no = 0

    while idx < len(plan) and budget.steps < max_steps:
        budget.tick()
        round_no += 1
        step = plan[idx]
        tool_raw: Any = step.get("tool")
        step_args_raw: Any = step.get("args")
        tool: str = tool_raw if isinstance(tool_raw, str) else str(tool_raw)
        args: Dict[str, Any] = dict(step_args_raw) if isinstance(step_args_raw, dict) else {}
        if args.get("flight_id") == "<best>":
            if best is None:
                # observation changed: best unknown -> replan now
                plan = _replan(failed, None, constraints)
                idx, since_replan, replans = 0, 0, replans + 1
                if verbose:
                    print(f"  [replan #{replans}: best unknown]")
                continue
            best_fid_raw: Any = best.get("flight_id")
            args["flight_id"] = best_fid_raw if isinstance(best_fid_raw, str) else ""
        if args.get("code") == "<from_book>":
            if not held_code:
                best_id_raw: Any = best.get("flight_id") if isinstance(best, dict) else None
                plan = _replan(
                    failed,
                    best_id_raw if isinstance(best_id_raw, str) else None,
                    constraints,
                )
                idx, since_replan, replans = 0, 0, replans + 1
                continue
            args["code"] = held_code

        # Narrow to Optional[Dict[str, Any]] so `check` never sees
        # a stray `str` (was: Dict | Any | str | None).
        flight_ctx: Optional[Dict[str, Any]] = None
        fid_h: Any = args.get("flight_id")
        if isinstance(fid_h, str):
            flight_ctx = _as_flight_dict(checked.get(fid_h))
        allowed, why = gate.check(tool, args, flight_ctx)
        if not allowed:
            handoff = build_handoff("NEEDS_APPROVAL", constraints, trace, tried,
                                    f"Approve {tool}({args})? {why}", side_effects)
            return AgentResult("NEEDS_APPROVAL", held_code,
                               f"Waiting approval: {why}", budget.steps,
                               trace, handoff)

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
            print(f"[V{round_no}] {tool}({args}) [{step.get('label')}] -> {obs.get('status')}")
        idx += 1
        since_replan += 1

        significant_change = False
        if tool == "search_flights" and obs.get("status") == "ok":
            flights_raw_h: Any = obs.get("flights", [])
            flights_h: List[Dict[str, Any]] = (
                [f for f in flights_raw_h if isinstance(f, dict)]
                if isinstance(flights_raw_h, list) else []
            )
            for f in flights_h:
                fid = f.get("flight_id")
                if isinstance(fid, str) and fid not in checked:
                    checked[fid] = f
            cands = [f for f in flights_h
                     if constraints.is_ok(f)
                     and isinstance(f.get("flight_id"), str)
                     and f.get("flight_id") not in failed
                     and isinstance(f.get("seats_left"), int) and int(f.get("seats_left", 0)) > 0]
            def _price_key(f: Dict[str, Any]) -> int:
                p = f.get("price")
                return p if isinstance(p, int) else 0
            new_best: Optional[Dict[str, Any]] = min(cands, key=_price_key) if cands else None
            if new_best is not None and (
                    best is None or new_best.get("flight_id") != best.get("flight_id")):
                significant_change = True
            best = new_best
        elif tool == "check_seat":
            if obs.get("status") == "ok":
                f_raw_h: Any = obs.get("flight")
                f_h: Dict[str, Any] = f_raw_h if isinstance(f_raw_h, dict) else {}
                fid_h2 = f_h.get("flight_id")
                if isinstance(fid_h2, str):
                    checked[fid_h2] = f_h
                if constraints.is_ok(f_h):
                    best_price_raw: Any = best.get("price") if isinstance(best, dict) else None
                    f_price_raw: Any = f_h.get("price")
                    if best is None or (
                            isinstance(f_price_raw, (int, float))
                            and isinstance(best_price_raw, (int, float))
                            and f_price_raw < best_price_raw):
                        best = f_h
            else:
                fail_h: Any = args.get("flight_id")
                if isinstance(fail_h, str):
                    failed.add(fail_h)
                significant_change = True  # expected flight gone -> replan
        elif tool == "book_seat":
            if obs.get("status") == "held":
                code_h: Any = obs.get("code")
                booking_h: Any = obs.get("booking")
                booking_d: Dict[str, Any] = booking_h if isinstance(booking_h, dict) else {}
                if isinstance(code_h, str):
                    held_code = code_h
                price_h: Any = booking_d.get("price")
                if isinstance(price_h, int):
                    seen_price = price_h
                side_effects.append(f"held {held_code}")
            else:
                fail_b: Any = args.get("flight_id")
                if isinstance(fail_b, str):
                    failed.add(fail_b)
                best = None
                significant_change = True
        elif tool == "pay" and obs.get("status") == "paid":
            paid = True
            code_p: Any = args.get("code")
            if isinstance(code_p, str):
                side_effects.append(f"paid {code_p}")

        done, msg = check_completion(held_code, constraints)
        if done:
            ok, cmsg = cross_check_price(held_code, seen_price)
            if ok:
                if verbose:
                    print(f"  [replans={replans}] Verified: {msg}")
                return AgentResult("SUCCESS", held_code,
                                   f"Hybrid done (replans={replans}), {msg}",
                                   budget.steps, trace)
            handoff = build_handoff("FAILED", constraints, trace, tried,
                                    f"Price mismatch: {cmsg}. Proceed?",
                                    side_effects)
            return AgentResult("FAILED", held_code, cmsg, budget.steps,
                               trace, handoff)

        best_id_for_loop: Any = best.get("flight_id") if isinstance(best, dict) else None
        flag = loop.check(tool, args, (best_id_for_loop if isinstance(best_id_for_loop, str) else None,
                                       held_code, paid))
        if flag == "LOOP":
            handoff = build_handoff("LOOP", constraints, trace, tried,
                                    f"Action {tool}({args}) repeated. Escalate?",
                                    side_effects)
            return AgentResult("LOOP", held_code, f"Loop at V{round_no}",
                               budget.steps, trace, handoff)
        if flag == "STALL":
            handoff = build_handoff("STALLED", constraints, trace, tried,
                                    "No progress across 5 rounds. New direction?",
                                    side_effects)
            return AgentResult("STALLED", held_code, f"Stall at V{round_no}",
                               budget.steps, trace, handoff)

        # Replan every k steps OR on significant observation change.
        if significant_change or since_replan >= k:
            if held_code is None and (significant_change or idx < len(plan)):
                replan_best_raw: Any = best.get("flight_id") if isinstance(best, dict) else None
                plan = _replan(
                    failed,
                    replan_best_raw if isinstance(replan_best_raw, str) else None,
                    constraints,
                )
                idx, since_replan, replans = 0, 0, replans + 1
                if verbose:
                    print(f"  [replan #{replans} after V{round_no}]")

        over, reason = budget.exceeded()
        if over:
            handoff = build_handoff("BUDGET_EXCEEDED", constraints, trace,
                                    tried, f"Budget hit ({reason}). Extend?",
                                    side_effects)
            return AgentResult("BUDGET_EXCEEDED", held_code, reason,
                               budget.steps, trace, handoff)

    done, msg = check_completion(held_code, constraints)
    if done:
        return AgentResult("SUCCESS", held_code, msg, budget.steps, trace)
    handoff = build_handoff("FAILED", constraints, trace, tried,
                            f"Hybrid ended without verified booking ({msg}). Replan?",
                            side_effects)
    return AgentResult("FAILED", held_code, msg, budget.steps, trace, handoff)


def build_hybrid_langchain_agent(model_name: str | None = None,
                                 temperature: float = 0.0):
    """LangChain variant: TodoListMiddleware gives plan+replan behaviour."""
    from typing import cast

    from langchain.agents import create_agent
    from langchain.agents.middleware import (
        ModelCallLimitMiddleware,
        TodoListMiddleware,
    )

    from models import get_chat_model

    model = get_chat_model(model_name, temperature=temperature)
    hybrid_prompt = (
        "Plan with write_todos, execute a few steps, then revise "
        "the todo list when observations change. "
        "Verify every booking with get_booking."
    )
    # TodoListMiddleware (PlanningState) and ModelCallLimitMiddleware
    # (ModelCallLimitState) use different State generics; LangChain merges
    # them at runtime. Cast to Any so the invariant StateT doesn't error.
    hybrid_middleware: Any = cast(
        Any,
        [TodoListMiddleware(),
         ModelCallLimitMiddleware(run_limit=14, exit_behavior="end")],
    )
    return create_agent(
        model=model, tools=TOOLS,
        system_prompt=hybrid_prompt,
        middleware=hybrid_middleware,
    )


if __name__ == "__main__":
    r = run_hybrid_task(scenario="standard", auto_approve=True)
    print(f"\nResult: {r.status} | {r.message} | steps={r.steps}")
