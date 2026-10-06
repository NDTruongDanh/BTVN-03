"""Run a REAL model (Ollama Cloud) against the flight-booking tools.

Usage:
  pip install -r requirements.txt
  cp .env.example .env   # then set OLLAMA_API_KEY + MODEL_NAME
  python run_live.py --pattern react --scenario standard
  python run_live.py --pattern plan --scenario standard
  python run_live.py --pattern hybrid --scenario dynamic_sold_out

MODEL_NAME formats (see models.py):
  ollama-cloud:gpt-oss:120b   Explicit Ollama Cloud (recommended)
  ollama:gpt-oss:120b         Cloud if OLLAMA_API_KEY is set, else local
  openai:gpt-4o-mini          Any other LangChain provider (passthrough)
"""

from __future__ import annotations

import argparse

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from agent_hybrid import build_hybrid_langchain_agent
from agent_plan_execute import build_plan_langchain_agent
from agent_react import SYSTEM_PROMPT as REACT_SYSTEM_PROMPT
from agent_react import build_react_langchain_agent
from flight_tools import reset_database
from harness import BookingConstraints
from models import describe_model_config


def build_task_prompt(constraints: BookingConstraints) -> str:
    return (
        f"Book a flight: origin={constraints.origin}, "
        f"destination={constraints.destination}, date={constraints.date}. "
        f"Constraints: depart strictly before {constraints.depart_before}, "
        f"price at most {constraints.max_price} VND. "
        "Rules: use tools for every fact, never invent flight IDs, prices, "
        "or booking codes. After pay, always call get_booking to verify. "
        "When done, reply with the booking code and price."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a live LangChain agent.")
    parser.add_argument("--pattern", choices=["react", "plan", "hybrid"],
                        default="react")
    parser.add_argument("--scenario", default="standard",
                        help="standard | dynamic_sold_out | over_budget | "
                             "timeout_loop | approval_only")
    parser.add_argument("--model", default=None,
                        help="Override MODEL_NAME env var for this run.")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    print(f"Model config: {describe_model_config(args.model)}")
    constraints = BookingConstraints()
    reset_database(args.scenario)

    if args.pattern == "react":
        agent = build_react_langchain_agent(args.model, args.temperature)
    elif args.pattern == "plan":
        agent = build_plan_langchain_agent(args.model, args.temperature)
    else:
        agent = build_hybrid_langchain_agent(args.model, args.temperature)

    prompt = build_task_prompt(constraints)
    print(f"Scenario: {args.scenario} | Pattern: {args.pattern}")
    print(f"Task: {prompt}\n")

    # System prompts are already baked into each builder; the user message
    # carries the concrete booking goal.
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    messages = result.get("messages", []) if isinstance(result, dict) else []

    for msg in messages:
        kind = type(msg).__name__
        text = getattr(msg, "text", None) or getattr(msg, "content", "")
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            for tc in tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", tc)
                call_args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", "")
                print(f"[{kind}] tool_call: {name}({call_args})")
        elif text:
            print(f"[{kind}] {text}")

    _ = REACT_SYSTEM_PROMPT  # keep import used if builders change prompts


if __name__ == "__main__":
    main()
