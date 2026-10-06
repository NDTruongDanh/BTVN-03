"""Offline evaluation: run all 3 patterns on the same scenario suite.

Scenarios:
  standard        - happy path, VN122 satisfies all constraints
  dynamic_sold_out- VN122 sells out after search (tests adaptability)
  over_budget     - all morning flights exceed budget (tests honest failure)
  timeout_loop    - check_seat always times out (tests loop detector)
  approval_only   - only non-refundable fare left (tests permission gate)

Metrics per run: status, steps, success. Prints a comparison table.
"""

from __future__ import annotations

from agent_hybrid import run_hybrid_task
from agent_plan_execute import run_plan_task
from agent_react import run_react_task
from harness import BookingConstraints

SCENARIOS = ["standard", "dynamic_sold_out", "over_budget",
             "timeout_loop", "approval_only"]


def main() -> None:
    constraints = BookingConstraints()
    print(f"Constraints: {constraints}\n")
    rows = []
    for sc in SCENARIOS:
        # approval_only must reach the human gate: do NOT auto-approve it.
        auto = (sc != "approval_only")
        r1 = run_react_task(constraints, scenario=sc, auto_approve=auto, verbose=False)
        r2 = run_plan_task(constraints, scenario=sc, auto_approve=auto, verbose=False)
        r3 = run_hybrid_task(constraints, scenario=sc, auto_approve=auto, verbose=False)
        rows.append((sc, r1, r2, r3))

    print(f"{'scenario':<18} {'ReAct':<22} {'Plan-then-Exec':<22} {'Hybrid':<22}")
    print("-" * 86)
    for sc, r1, r2, r3 in rows:
        print(f"{sc:<18} {r1.status + f'({r1.steps})':<22} "
              f"{r2.status + f'({r2.steps})':<22} {r3.status + f'({r3.steps})':<22}")

    print("\nSuccess count:")
    for name, idx in (("ReAct", 1), ("Plan", 2), ("Hybrid", 3)):
        ok = sum(1 for _, *rs in rows if rs[idx - 1].status == "SUCCESS")
        print(f"  {name}: {ok}/{len(rows)}")


if __name__ == "__main__":
    main()
