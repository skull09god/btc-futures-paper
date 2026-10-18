
"""Initial smoke tests for the BTCUSDT paper-trading engine."""
import os
import sys
import pytest

# Import the actual engine in this project.
sys.path.insert(0, os.path.dirname(__file__))
import btc_futures_paper as E


@pytest.fixture
def tmp_env(tmp_path, monkeypatch):
    """Keep tests isolated from the real paper-trading state and journal."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(E, "STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.setattr(E, "JOURNAL_FILE", str(tmp_path / "journal.csv"))
    monkeypatch.setattr(E, "LOCK_FILE", str(tmp_path / "lock"))
    monkeypatch.setattr(E, "LOG_FILE", str(tmp_path / "log.txt"))
    yield tmp_path


def test_engine_is_paper_only():
    assert E.PAPER_ONLY is True
    E._assert_paper_only()


def test_initial_state_passes_validation(tmp_env):
    state = E.initial_state()
    E.validate_state(state)


def test_contradictory_position_and_pending_are_rejected(tmp_env):
    state = E.initial_state()
    state["position"] = {
        "trade_id": "T1", "side": "LONG", "entry_price": 100.0,
        "stop": 99.0, "target": 102.0, "quantity_btc": 0.1,
        "estimated_risk_inr": 100.0, "entry_fee_inr": 5.0,
        "entry_time": "2026-10-18T00:00:00+00:00",
        "fill_bar": 0, "last_checked_bar": 0,
    }
    state["pending"] = {
        "pending_id": "P1", "side": "LONG", "entry": 100.0,
        "stop": 99.0, "target": 102.0, "quantity_btc": 0.1,
        "estimated_risk_inr": 100.0, "estimated_net_rr": 2.1,
        "bars_waited": 0, "created_at": "2026-10-18T00:00:00+00:00",
        "signal_bar": 0,
    }
    with pytest.raises(E.StateError):
        E.validate_state(state)


def test_invalid_ohlc_is_rejected():
    bad_row = [0, "100", "99", "100.5", "100", "10", 299]
    with pytest.raises(E.DataError):
        E._validate_kline_row(bad_row)


def test_disabled_paper_only_mode_exits(monkeypatch):
    monkeypatch.setattr(E, "PAPER_ONLY", False)
    with pytest.raises(SystemExit):
        E._assert_paper_only()


# ---------- risk and trade-execution regression tests ----------

def test_sizing_enforces_risk_cap(tmp_env):
    state = E.initial_state()
    sig = {
        "side": "LONG",
        "entry": 50000.0,
        "stop": 49500.0,
        "target": 51300.0,
        "reason": "test",
        "signal_bar": 0,
    }

    size, reason = E.size_position(sig, state, 83.0)

    assert size is not None, reason
    assert size["estimated_risk_inr"] <= E.RISK_CAP_INR + 1e-6


def test_sizing_rejects_insufficient_net_rr(tmp_env):
    state = E.initial_state()
    sig = {
        "side": "LONG",
        "entry": 50000.0,
        "stop": 49500.0,
        "target": 50400.0,
        "reason": "test",
        "signal_bar": 0,
    }

    size, reason = E.size_position(sig, state, 83.0)

    assert size is None


def test_close_position_is_idempotent(tmp_env):
    state = E.initial_state()
    state["position"] = {
        "trade_id": "T-IDEMPOTENT",
        "side": "LONG",
        "entry_price": 50000.0,
        "stop": 49500.0,
        "target": 51100.0,
        "quantity_btc": 0.001,
        "estimated_risk_inr": 500.0,
        "entry_fee_inr": 2.0,
        "entry_time": "2026-10-18T00:00:00+00:00",
        "fill_bar": 0,
        "last_checked_bar": 0,
        "reason": "regression test",
    }

    E.close_position(state, 83.0, 51100.0, "TAKE_PROFIT")
    balance_after_first_close = state["balance_inr"]

    E.close_position(state, 83.0, 51100.0, "TAKE_PROFIT")

    assert state["balance_inr"] == pytest.approx(balance_after_first_close)


def test_stop_loss_wins_when_stop_and_target_are_both_touched(tmp_env):
    state = E.initial_state()
    state["position"] = {
        "trade_id": "T-STOP-PRIORITY",
        "side": "LONG",
        "entry_price": 50000.0,
        "stop": 49500.0,
        "target": 51300.0,
        "quantity_btc": 0.001,
        "estimated_risk_inr": 500.0,
        "entry_fee_inr": 2.0,
        "entry_time": "2026-10-18T00:00:00+00:00",
        "fill_bar": 0,
        "last_checked_bar": 0,
        "reason": "regression test",
    }

    bar = {
        "open_time": E.BAR_MS_5M,
        "open": 50000.0,
        "high": 51200.0,
        "low": 49400.0,
        "close": 50100.0,
        "volume": 100.0,
        "close_time": 2 * E.BAR_MS_5M - 1,
    }

    E.bar_exit_check(state, bar, 83.0)

    assert state["position"] is None
    rows, _ = E.read_journal()
    closed = [
        row for row in rows
        if row.get("event") == "POSITION_CLOSED"
        and row.get("trade_id") == "T-STOP-PRIORITY"
    ]
    assert closed, "Expected a POSITION_CLOSED journal event"
    assert closed[-1]["reason"] == "STOP_LOSS"


def test_startup_reconcile_applies_missing_close(tmp_env):
    state = E.initial_state()
    state["position"] = {
        "trade_id": "T-RECOVERY",
        "side": "LONG",
        "entry_price": 50000.0,
        "stop": 49500.0,
        "target": 51300.0,
        "quantity_btc": 0.001,
        "estimated_risk_inr": 500.0,
        "entry_fee_inr": 2.0,
        "entry_time": "2026-10-18T00:00:00+00:00",
        "fill_bar": 0,
        "last_checked_bar": 0,
        "reason": "recovery test",
    }

    E.append_journal(
        "ENTRY_FILLED",
        trade_id="T-RECOVERY",
        side="LONG",
        entry=50000.0,
        stop=49500.0,
        target=51300.0,
        quantity_btc=0.001,
        risk_inr=500.0,
        fees_inr=2.0,
        fx_rate=83.0,
        reason="recovery test",
        balance_inr=1000000.0,
        equity_inr=1000000.0,
    )
    E.append_journal(
        "POSITION_CLOSED",
        trade_id="T-RECOVERY",
        side="LONG",
        entry=50000.0,
        stop=49500.0,
        target=51100.0,
        quantity_btc=0.001,
        exit_price=51300.0,
        fx_rate=83.0,
        risk_inr=500.0,
        gross_pnl_inr=91.3,
        fees_inr=4.0,
        net_pnl_inr=87.3,
        reason="TAKE_PROFIT",
        balance_inr=1000087.3,
        equity_inr=1000087.3,
    )

    rows, _ = E.read_journal()
    repaired, flagged, reason = E.reconcile_state_with_journal(
        state, rows
    )

    assert flagged, "Expected the current implementation to flag a journaled close"
    assert "POSITION_CLOSED" in reason
    assert repaired["position"] is None
    assert repaired["balance_inr"] > 1000000.0

def test_recovery_uses_journaled_net_pnl(tmp_env):
    state = E.initial_state()
    state["position"] = {
        "trade_id": "T-RECOVERY-PNL",
        "side": "LONG",
        "entry_price": 50000.0,
        "stop": 49500.0,
        "target": 51300.0,
        "quantity_btc": 0.001,
        "estimated_risk_inr": 500.0,
        "entry_fee_inr": 2.0,
        "entry_time": "2026-10-18T00:00:00+00:00",
        "fill_bar": 0,
        "last_checked_bar": 0,
        "reason": "recovery consistency test",
    }

    E.append_journal(
        "ENTRY_FILLED",
        trade_id="T-RECOVERY-PNL",
        side="LONG",
        entry=50000.0,
        stop=49500.0,
        target=51300.0,
        quantity_btc=0.001,
        risk_inr=500.0,
        fees_inr=2.0,
        fx_rate=83.0,
        reason="recovery consistency test",
        balance_inr=1000000.0,
        equity_inr=1000000.0,
    )

    expected_net_pnl = 123.45

    E.append_journal(
        "POSITION_CLOSED",
        trade_id="T-RECOVERY-PNL",
        side="LONG",
        entry=50000.0,
        stop=49500.0,
        target=51300.0,
        quantity_btc=0.001,
        exit_price=51100.0,
        fx_rate=83.0,
        risk_inr=500.0,
        gross_pnl_inr=130.0,
        fees_inr=6.55,
        net_pnl_inr=expected_net_pnl,
        reason="TAKE_PROFIT",
        balance_inr=1000000.0 + expected_net_pnl,
        equity_inr=1000000.0 + expected_net_pnl,
    )

    rows, errors = E.read_journal()
    assert not errors

    repaired, flagged, reason = E.reconcile_state_with_journal(state, rows)

    assert flagged
    assert "POSITION_CLOSED" in reason
    assert repaired["position"] is None
    assert repaired["balance_inr"] == pytest.approx(
        1000000.0 + expected_net_pnl
    )

def test_recovery_does_not_apply_close_twice(tmp_env):
    state = E.initial_state()
    state["position"] = {
        "trade_id": "T-RECOVERY-ONCE",
        "side": "LONG",
        "entry_price": 50000.0,
        "stop": 49500.0,
        "target": 51300.0,
        "quantity_btc": 0.001,
        "estimated_risk_inr": 500.0,
        "entry_fee_inr": 2.0,
        "entry_time": "2026-10-18T00:00:00+00:00",
        "fill_bar": 0,
        "last_checked_bar": 0,
        "reason": "idempotency test",
    }

    E.append_journal(
        "ENTRY_FILLED",
        trade_id="T-RECOVERY-ONCE",
        side="LONG",
        entry=50000.0,
        stop=49500.0,
        target=51300.0,
        quantity_btc=0.001,
        risk_inr=500.0,
        fees_inr=2.0,
        fx_rate=83.0,
        reason="idempotency test",
        balance_inr=1000000.0,
        equity_inr=1000000.0,
    )

    net_pnl = 123.45
    E.append_journal(
        "POSITION_CLOSED",
        trade_id="T-RECOVERY-ONCE",
        side="LONG",
        entry=50000.0,
        stop=49500.0,
        target=51300.0,
        quantity_btc=0.001,
        exit_price=51100.0,
        fx_rate=83.0,
        risk_inr=500.0,
        gross_pnl_inr=130.0,
        fees_inr=6.55,
        net_pnl_inr=net_pnl,
        reason="TAKE_PROFIT",
        balance_inr=1000000.0 + net_pnl,
        equity_inr=1000000.0 + net_pnl,
    )

    rows, errors = E.read_journal()
    assert not errors

    repaired, flagged, reason = E.reconcile_state_with_journal(state, rows)
    assert flagged
    assert repaired["position"] is None
    first_balance = repaired["balance_inr"]
    assert first_balance == pytest.approx(1000000.0 + net_pnl)
        # Simulate a process restart by reloading the saved state from disk.
    E.save_state(repaired)
    reloaded = E.load_state()

    assert reloaded["position"] is None
    assert reloaded["balance_inr"] == pytest.approx(first_balance)
    assert reloaded["needs_recovery"] is True
    assert reloaded["halted_today"] is True

    # Simulate a second reconciliation after the repaired state is saved.
    second_state, second_flagged, second_reason = (
        E.reconcile_state_with_journal(repaired, rows)
    )

    assert second_state["balance_inr"] == pytest.approx(first_balance)
    assert second_state["position"] is None


@pytest.mark.parametrize(
    "bad_pnl",
    [
        None,
        "",
        "not-a-number",
        "NaN",
        "Infinity",
        "-Infinity",
    ],
)
def test_journal_net_pnl_rejects_invalid_values(bad_pnl):
    row = {"net_pnl_inr": bad_pnl}

    with pytest.raises(E.StateError):
        E._journal_net_pnl(row)


def test_journal_net_pnl_accepts_finite_value():
    row = {"net_pnl_inr": "123.45"}

    assert E._journal_net_pnl(row) == pytest.approx(123.45)


def test_read_journal_rejects_missing_header_columns(tmp_path, monkeypatch):
    journal = tmp_path / "journal.csv"
    journal.write_text(
        "event_id,event\n"
        "evt-1,POSITION_CLOSED\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(E, "JOURNAL_FILE", str(journal))

    rows, errors = E.read_journal()

    assert errors, "Malformed journal header should be rejected"
    assert rows == []
