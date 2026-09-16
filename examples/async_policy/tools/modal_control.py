"""Bound the entire Modal request, including retries, and confirm app cleanup."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from examples.async_policy.tools.campaign_state import BudgetLedger, exclusive_lock, write_json


def stop_owned_app(app_id: str | None, name: str) -> dict:
    """Stop only this request's uniquely named app; never touch unrelated apps."""
    prefix = [sys.executable, "-m", "modal", "app"]
    errors = []
    try:
        listing = subprocess.check_output([*prefix, "list", "--json"], text=True, timeout=30)
        owned = [row for row in json.loads(listing)
                 if row.get("app_id") == app_id or row.get("description") == name]
        for row in owned:
            if row.get("state") != "stopped" or int(row.get("tasks", 0)):
                result = subprocess.run([*prefix, "stop", "--yes", row["app_id"]], capture_output=True,
                                        text=True, timeout=30)
                if result.returncode:
                    errors.append(result.stderr[-2000:])
        for _ in range(3):
            listing = subprocess.check_output([*prefix, "list", "--json"], text=True, timeout=30)
            active = [row for row in json.loads(listing)
                      if (row.get("app_id") == app_id or row.get("description") == name)
                      and (row.get("state") != "stopped" or int(row.get("tasks", 0)))]
            if not active:
                return {"confirmed_stopped": True, "app_id": app_id, "errors": errors}
            time.sleep(1)
        return {"confirmed_stopped": False, "app_id": app_id, "active": active, "errors": errors}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"confirmed_stopped": False, "app_id": app_id, "errors": [*errors, str(exc)]}


def invoke(app, remote, arguments: tuple, *, output: Path, budget: BudgetLedger,
           timeout_s: float, gpus: int, app_name: str) -> dict:
    """Reserve before provisioning and keep an unknown attempt fully charged."""
    with exclusive_lock(budget.path.with_suffix(".run.lock")):
        if budget.summary()["unconfirmed_attempts"]:
            raise RuntimeError("previous paid attempt has unconfirmed cleanup; reconcile it before launching another")
        reservation_id = uuid.uuid4().hex
        reservation = budget.reserve(reservation_id, timeout_s + 120, gpus)
        started = time.monotonic()
        deadline = time.time() + timeout_s
        app_id, call, result = None, None, None
        failure = None
        request_path = output / "request.json"
        request = json.loads(request_path.read_text())
        request.update(budget_attempt_id=reservation_id, budget_reservation=reservation,
                       deadline_epoch=deadline, app_name=app_name)
        write_json(request_path, request)
        previous_handler = signal.getsignal(signal.SIGALRM)

        def expired(signum, frame):
            raise TimeoutError("Modal controller deadline expired")

        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
        try:
            with app.run():
                app_id = app.app_id
                request["app_id"] = app_id
                write_json(request_path, request)
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError("campaign deadline expired during provisioning")
                call = remote.spawn(*arguments, deadline)
                request["function_call_id"] = call.object_id
                write_json(request_path, request)
                try:
                    result = call.get(timeout=max(0.001, deadline - time.time()))
                except BaseException:
                    call.cancel(terminate_containers=True)
                    raise
        except BaseException as exc:
            failure = exc
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            cleanup = stop_owned_app(app_id, app_name)
            elapsed = time.monotonic() - started
            budget.finish(reservation_id, elapsed, confirmed_stopped=cleanup["confirmed_stopped"],
                          status="failed" if failure else "returned")
            write_json(output / "cleanup.json", cleanup)
            write_json(output / "cost.json", {"elapsed_s": elapsed, "reservation_id": reservation_id,
                                               "budget": budget.summary()})
        if failure is not None:
            raise failure
        if not cleanup["confirmed_stopped"]:
            raise RuntimeError("Modal app cleanup could not be confirmed; paid execution is paused")
        return result
