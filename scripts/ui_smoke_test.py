#!/usr/bin/env python3
"""Headless Streamlit smoke test for the Phase 2 Command Center.

Runs the real app with Streamlit's AppTest harness:
Home -> Command -> 30% stock + 0.6x capacity -> counterfactual.

    python scripts/ui_smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest


def main() -> int:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    if at.exception:
        for exc in at.exception:
            print(f"FAIL home: {exc.value}", file=sys.stderr)
        return 1

    command = next((button for button in at.button if button.label == "Command"), None)
    if command is None:
        print("FAIL: Command nav button not found", file=sys.stderr)
        return 1
    command.click()
    at.run()
    if at.exception:
        for exc in at.exception:
            print(f"FAIL command: {exc.value}", file=sys.stderr)
        return 1

    by_label = {slider.label: slider for slider in at.slider}
    by_label["Available regional stock"].set_value(30)
    by_label["Bed capacity multiplier"].set_value(0.6)
    at.run()
    if at.exception:
        for exc in at.exception:
            print(f"FAIL scarcity stress: {exc.value}", file=sys.stderr)
        return 1
    if not at.error:
        print("FAIL: expected a critical deficit alert at 30% stock", file=sys.stderr)
        return 1

    conservative = next(
        (toggle for toggle in at.toggle if toggle.label == "Plan against the 80% upper interval"),
        None,
    )
    if conservative is not None:
        conservative.set_value(True)
        at.run()
        if at.exception:
            for exc in at.exception:
                print(f"FAIL conservative mode: {exc.value}", file=sys.stderr)
            return 1

    run = next((button for button in at.button if button.label == "Run counterfactual"), None)
    if run is None:
        print("FAIL: simulator button not found", file=sys.stderr)
        return 1
    run.click()
    at.run()
    if at.exception:
        for exc in at.exception:
            print(f"FAIL simulator: {exc.value}", file=sys.stderr)
        return 1
    if not any("ROI" in str(block.value) or "Intervention" in str(block.value) for block in at.markdown):
        print("WARN: ROI heading not detected", file=sys.stderr)

    assistant = next((button for button in at.button if button.label == "Assistant"), None)
    if assistant is None:
        print("FAIL: Assistant nav button not found", file=sys.stderr)
        return 1
    assistant.click()
    at.run()
    if at.exception:
        for exc in at.exception:
            print(f"FAIL assistant page: {exc.value}", file=sys.stderr)
        return 1
    if not at.chat_input:
        print("FAIL: assistant chat input not found", file=sys.stderr)
        return 1
    at.chat_input[0].set_value("Why is my ward short?").run()
    if at.exception:
        for exc in at.exception:
            print(f"FAIL assistant answer: {exc.value}", file=sys.stderr)
        return 1

    role = next((box for box in at.selectbox if box.label == "View as"), None)
    if role is not None:
        role.set_value("Hospital surge")
        at.run()
        at.chat_input[0].set_value("What should I do first?").run()
        if at.exception:
            for exc in at.exception:
                print(f"FAIL assistant role switch: {exc.value}", file=sys.stderr)
            return 1

    print(
        "PASS — Home, Command Center, 30% stress, counterfactual and Assistant "
        "chat render without exceptions."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
