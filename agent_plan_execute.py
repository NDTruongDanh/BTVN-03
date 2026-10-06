"""Plan-then-Execute agent: one model call makes the FULL plan, then execute blindly.

Pros: plan is visible up front (reviewable, cost-estimable).
Cons: an early-step error poisons everything after it; no adaptation.

Also provides build_plan_langchain_agent() showing the LangChain
HumanInTheLoopMiddleware approval gate for the risky book/pay steps.
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
    PermissionChecker,
    TraceLogger,
    build_handoff,
    check_completion,
)

EXECUTORS = {
    "search_flights": raw_search_flights,
    "check_seat": raw_check_seat,
    "book_seat": raw_book_seat,
    "pay": raw_pay,
    "get_booking": raw_get_booking,
}


def make_plan(constraints: BookingConstraints) -> List[Dict[str, Any]]:
    """Single planning call: fixed 5-step skeleton (the 'plan' artifact)."""
    return [
        {"tool": "search_flights",
         "args": {"origin": constraints.origin,
                  "destination": constraints.destination,
                  "date": constraints.date}},
        {"tool": "check_seat", "args": {"flight_id": "VN122"}},
        {"tool": "book_seat", "args": {"flight_id": "VN122"}},
        {"tool": "pay", "args": {"code": "<from_book>"}},
        {"tool": "get_booking", "args": {"code": "<from_book>"}},
    ]


def run_plan_task(constraints: BookingConstraints | None = None,
                  scenario: str = "standard",
                  max_steps: int = 12,
                  auto_approve: bool = False,
                  approve_plan: bool = True,
                  verbose: bool = True) -> AgentResult:
    constraints = constraints or BookingConstraints()
    reset_database(scenario)
    trace = TraceLogger()
    budget = BudgetTracker(max_steps=max_steps)
    gate = PermissionChecker(auto_approve=auto_approve)
    tried: List[str] = []
    side_effects: List[str] = []
    held_code: Optional[str] = None

    plan = make_plan(constraints)
    if verbose:
        print("PLAN:")
        for i, s in enumerate(plan, 1):
            print(f"  {i}. {s['tool']}({s['args']})")
    if not approve_plan:  # human rejected the plan
        handoff = build_handoff("NEEDS_APPROVAL", constraints, trace, tried,
                                "Plan rejected by reviewer. Provide a new plan?",
                                side_effects)
        return AgentResult("NEEDS_APPROVAL", None, "plan rejected",
                           0, trace, handoff)

    for round_no, step in enumerate(plan, 1):
        budget.tick()
        tool_raw: Any = step.get("tool")
        step_args_raw: Any = step.get("args")
        tool: str = tool_raw if isinstance(tool_raw, str) else str(tool_raw)
        args: Dict[str, Any] = dict(step_args_raw) if isinstance(step_args_raw, dict) else {}
        # resolve placeholder from the earlier book step
        if args.get("code") == "<from_book>":
            if not held_code:
                handoff = build_handoff("FAILED", constraints, trace, tried,
                                        "Plan broken: pay step has no booking code "
                                        "because booking failed. Replan?",
                                        side_effects)
                return AgentResult("FAILED", None,
                                   "plan broken: no booking code for pay",
                                   budget.steps, trace, handoff)
            args["code"] = held_code

        # checklist #0: permission BEFORE execution
        allowed, why = gate.check(tool, args)
        if not allowed:
            handoff = build_handoff("NEEDS_APPROVAL", constraints, trace,
                                    tried,
                                    f"Approve {tool}({args})? {why}",
                                    side_effects)
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
            print(f"[V{round_no}] {tool}({args}) -> {obs}")

        # Plan-then-execute does NOT replan: any non-ok observation aborts.
        ok_status = {"ok", "held", "paid"}
        if obs.get("status") not in ok_status:
            handoff = build_handoff("FAILED", constraints, trace, tried,
                                    f"Step {round_no} {tool} failed "
                                    f"({obs.get('status')}: {obs.get('hint', obs)}). "
                                    "Plan cannot continue without replanning. New plan?",
                                    side_effects)
            return AgentResult("FAILED", held_code,
                               f"plan step {round_no} failed: {obs}",
                               budget.steps, trace, handoff)
        if tool == "book_seat":
            code_raw: Any = obs.get("code")
            if isinstance(code_raw, str):
                held_code = code_raw
                side_effects.append(f"held {held_code}")
        if tool == "pay":
            pay_code: Any = args.get("code")
            if isinstance(pay_code, str):
                side_effects.append(f"paid {pay_code}")

    done, msg = check_completion(held_code, constraints)
    if done:
        return AgentResult("SUCCESS", held_code, f"Plan done, {msg}",
                           budget.steps, trace)
    # Blind execution may finish yet violate constraints (e.g. wrong flight).
    handoff = build_handoff("FAILED", constraints, trace, tried,
                            f"Plan executed but goal unverified ({msg}). Replan?",
                            side_effects)
    return AgentResult("FAILED", held_code, msg, budget.steps, trace, handoff)


def build_plan_langchain_agent(model_name: str | None = None,
                               temperature: float = 0.0):
    """LangChain variant with a human approval gate on book/pay."""
    from typing import cast

    from langchain.agents import create_agent
    from langchain.agents.middleware import (
        HumanInTheLoopMiddleware,
        ModelCallLimitMiddleware,
    )

    from models import get_chat_model

    model = get_chat_model(model_name, temperature=temperature)
    first_tool_desc: str = (
        str(TOOLS[0].description) if TOOLS and hasattr(TOOLS[0], "description") else ""
    )
    # HumanInTheLoopMiddleware is generic over StateT while
    # ModelCallLimitMiddleware pins ModelCallLimitState; LangChain merges
    # them at runtime. Cast to Any so the invariant StateT doesn't error.
    plan_middleware: Any = cast(
        Any,
        [
            HumanInTheLoopMiddleware(interrupt_on={"book_seat": True, "pay": True}),
            ModelCallLimitMiddleware(run_limit=12, exit_behavior="end"),
        ],
    )
    return create_agent(
        model=model,
        tools=TOOLS,
        system_prompt="First output a numbered plan, wait for approval, "
                      "then execute it step by step without deviation. " + first_tool_desc,
        middleware=plan_middleware,
    )


if __name__ == "__main__":
    r = run_plan_task(scenario="standard", auto_approve=True)
    print(f"\nResult: {r.status} | {r.message} | steps={r.steps}")
