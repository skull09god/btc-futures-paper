"""
BTCUSDT Perpetual Futures PAPER-TRADING ENGINE v3.0
====================================================
PAPER TRADING ONLY — no live endpoints, no API keys, no real orders.

Strategy preserved verbatim from v1.1:
  15m EMA50 trend+slope  |  5m EMA9/EMA21 pullback+reclaim+breakout
  ATR% 0.10–1.20         |  Volume >= 0.80 × 20-bar avg
  5-bar swing stop ±0.15 ATR, distance 0.80–2.50 ATR
  Target 2.2R, net R:R >= 2.0 after costs
  3-bar pending expiry, stop wins ties, fill-candle target NOT credited

Risk limits preserved:
  ₹10,00,000 start | ₹5,000 risk cap | 2 fills/UTC day
  ₹30,000 daily loss (equity-based) | 3 consecutive-loss halt

Reliability engineering (new):
  * Journal-first durable financial transitions, state-second.
  * Startup reconciliation: journal vs state; auto-repair safe cases,
    flag `needs_recovery` for unsafe contradictions.
  * Chronological per-candle processing with durable checkpoint advance.
  * Candle/fx validation, gap detection, staleness checks.
  * Process lock via fcntl (Unix). Atomic tmp+fsync+rename+parent-fsync.
  * Exhaustive state schema validation; corrupt state -> backup + stop.

Fill model documented as assumption (see `fill_pending`).
Margin/liquidation NOT modelled; this is a P&L simulator, not an
exchange replica.
"""

import csv
import json
import logging
import math
import os
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone

import requests

try:
    import fcntl  # POSIX
except ImportError:
    fcntl = None


# ============================================================ CONFIG

SYMBOL = "BTCUSDT"
BASE_URL = "https://fapi.binance.com"

STARTING_BALANCE_INR = 1_000_000.0
RISK_CAP_INR = 5_000.0
MAX_TRADES_PER_DAY = 2
DAILY_LOSS_LIMIT_INR = 30_000.0
MAX_CONSECUTIVE_LOSSES = 3

FEE_PER_SIDE = 0.0005
SLIPPAGE_PER_SIDE = 0.0002

TARGET_R_MULTIPLE = 2.2
MIN_NET_RR = 2.0

POLL_SECONDS = 10
KLINE_LIMIT = 120
PENDING_EXPIRY_BARS = 3

BAR_MS_5M = 5 * 60 * 1000
BAR_MS_15M = 15 * 60 * 1000
GAP_TOLERANCE_MS = 60 * 1000

FX_MAX_CACHE_AGE_S = 1800
FX_MIN, FX_MAX = 50.0, 200.0

SCHEMA_VERSION = 3
STATE_FILE = "btc_paper_state.json"
JOURNAL_FILE = "btc_paper_journal.csv"
LOG_FILE = "btc_paper_engine.log"
LOCK_FILE = "btc_paper_engine.lock"

PAPER_ONLY = True  # asserted at startup; engine refuses to run if False

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "BTC-Paper-Engine/3.0"})

logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


# ============================================================ EXC

class RateLimited(Exception):
    def __init__(self, retry_after):
        super().__init__(f"Rate limited; retry after {retry_after}s")
        self.retry_after = retry_after


class StateError(RuntimeError):
    """State/journal invalid or contradictory; engine must halt."""


class DataError(RuntimeError):
    """Candle/FX/mark data invalid; caller must flag needs_recovery."""


# ============================================================ UTIL

def utc_now():
    return datetime.now(timezone.utc)


def utc_day():
    return utc_now().date().isoformat()


def log(msg):
    try:
        print(f"[{utc_now().isoformat(timespec='seconds')}] {msg}")
    except Exception:
        pass
    logging.info(msg)


def new_id(prefix):
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)

def _journal_net_pnl(row):
    """Return validated recorded P&L or stop unsafe journal recovery."""
    raw = row.get("net_pnl_inr")

    if raw is None or not str(raw).strip():
        raise StateError(
            "Journal POSITION_CLOSED missing net_pnl_inr; "
            "manual recovery required."
        )

    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise StateError(
            "Journal POSITION_CLOSED has invalid net_pnl_inr; "
            "manual recovery required."
        )

    if not _finite(value):
        raise StateError(
            "Journal POSITION_CLOSED has non-finite net_pnl_inr; "
            "manual recovery required."
        )

    return value



# ============================================================ LOCK

class InstanceLock:
    """POSIX flock; no-op on non-POSIX with a warning."""

    def __init__(self, path):
        self.path = path
        self.fd = None

    def acquire(self):
        if fcntl is None:
            log("WARN: fcntl unavailable; process lock disabled. "
                "Do not run multiple engine instances.")
            return True
        self.fd = open(self.path, "w")
        try:
            fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.fd.write(str(os.getpid()))
            self.fd.flush()
            return True
        except (BlockingIOError, OSError):
            try:
                self.fd.close()
            except Exception:
                pass
            self.fd = None
            return False

    def release(self):
        if self.fd is not None and fcntl is not None:
            try:
                fcntl.flock(self.fd.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                self.fd.close()
            except Exception:
                pass
            self.fd = None


# ============================================================ HTTP

def _retry_after(resp):
    try:
        return int(resp.headers.get("Retry-After", 30))
    except (TypeError, ValueError):
        return 30


def get_json(path, params=None):
    r = SESSION.get(BASE_URL + path, params=params, timeout=12)
    if r.status_code in (418, 429):
        raise RateLimited(min(max(_retry_after(r), 1), 300))
    r.raise_for_status()
    try:
        return r.json()
    except ValueError as exc:
        raise DataError(f"Malformed JSON from {path}: {exc}")


def _validate_kline_row(row):
    if not isinstance(row, (list, tuple)) or len(row) < 7:
        raise DataError(f"Malformed kline row: {row!r}")
    try:
        o, h, l, c, v = (float(row[1]), float(row[2]), float(row[3]),
                         float(row[4]), float(row[5]))
    except (TypeError, ValueError):
        raise DataError(f"Non-numeric OHLCV: {row!r}")
    for x in (o, h, l, c):
        if not _finite(x) or x <= 0:
            raise DataError(f"Non-positive/non-finite price: {row[:5]!r}")
    if not _finite(v) or v < 0:
        raise DataError(f"Invalid volume: {row!r}")
    if h < l or h < o or h < c or l > o or l > c:
        raise DataError(f"Impossible OHLC: {row[:5]!r}")
    return o, h, l, c, v


def get_klines(interval, limit=KLINE_LIMIT):
    if interval not in ("5m", "15m"):
        raise DataError(f"Unsupported interval: {interval!r}")
    raw = get_json("/fapi/v1/klines",
                   {"symbol": SYMBOL, "interval": interval, "limit": limit})
    if not isinstance(raw, list):
        raise DataError(f"Klines not a list: {type(raw).__name__}")
    if not raw:
        raise DataError("Empty klines response")

    now_ms = int(time.time() * 1000)
    bars, seen = [], set()
    for row in raw:
        o, h, l, c, v = _validate_kline_row(row)
        ot, ct = int(row[0]), int(row[6])
        if ct <= ot:
            raise DataError(f"close_time<=open_time in row {ot}")
        if ot in seen:
            raise DataError(f"Duplicate candle open_time={ot}")
        seen.add(ot)
        if ot > now_ms + BAR_MS_15M:
            raise DataError(f"Future-dated candle open_time={ot}")
        bars.append({"open_time": ot, "open": o, "high": h, "low": l,
                     "close": c, "volume": v, "close_time": ct})

    bars.sort(key=lambda b: b["open_time"])
    bars = [b for b in bars if b["close_time"] < now_ms]
    if not bars:
        raise DataError("No completed candles after filtering")

    step = BAR_MS_5M if interval == "5m" else BAR_MS_15M
    for i in range(1, len(bars)):
        delta = bars[i]["open_time"] - bars[i - 1]["open_time"]
        if delta > step + GAP_TOLERANCE_MS:
            raise DataError(
                f"Gap in {interval}: {bars[i-1]['open_time']} -> "
                f"{bars[i]['open_time']} delta={delta}ms"
            )
        if delta < step - GAP_TOLERANCE_MS:
            raise DataError(
                f"Overlapping {interval} candles: delta={delta}ms"
            )

    latest = bars[-1]["open_time"]
    expected_latest = (now_ms // step) * step - step
    if latest < expected_latest - step:
        raise DataError(
            f"Stale {interval} history: latest={latest}, expected>="
            f"{expected_latest - step}"
        )
    return bars


def get_mark_price():
    data = get_json("/fapi/v1/premiumIndex", {"symbol": SYMBOL})
    try:
        price = float(data["markPrice"])
    except (KeyError, TypeError, ValueError):
        raise DataError(f"Bad premiumIndex payload: {data!r}")
    if not _finite(price) or not (0 < price < 10_000_000):
        raise DataError(f"Implausible mark price: {price}")
    return price


def get_usd_inr(last_known=None):
    try:
        r = SESSION.get("https://open.er-api.com/v6/latest/USD", timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("result") == "success":
            rate = float(data["rates"]["INR"])
            if FX_MIN < rate < FX_MAX:
                return rate
    except Exception as exc:
        logging.warning(f"FX fetch warning: {exc}")
    if last_known is not None and FX_MIN < last_known < FX_MAX:
        return float(last_known)
    raise DataError("FX unavailable and no valid fallback.")


def refresh_fx(state):
    now = time.time()
    cached = state.get("last_fx_rate")
    age = now - float(state.get("last_fx_fetch_time") or 0)
    if cached is None or age > FX_MAX_CACHE_AGE_S:
        rate = get_usd_inr(cached)
        if not (FX_MIN < rate < FX_MAX) or not _finite(rate):
            raise DataError(f"Implausible FX rate: {rate}")
        state["last_fx_rate"] = rate
        state["last_fx_fetch_time"] = now
        return rate, True
    if not (FX_MIN < cached < FX_MAX) or not _finite(cached):
        raise DataError(f"Cached FX implausible: {cached}")
    return cached, False


# ============================================================ INDICATORS

def ema_series(values, period):
    if len(values) < period:
        return []
    a = 2.0 / (period + 1.0)
    out = [values[0]]
    for v in values[1:]:
        out.append(a * v + (1 - a) * out[-1])
    return out


def ema(values, period):
    s = ema_series(values, period)
    return s[-1] if s else None


def atr(bars, period=14):
    if len(bars) < period + 1:
        return None
    trs = []
    for i in range(1, len(bars)):
        h, l, pc = bars[i]["high"], bars[i]["low"], bars[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs[-period:]) / period


# ============================================================ JOURNAL

JOURNAL_COLUMNS = [
    "timestamp_utc", "event_id", "trade_id", "pending_id", "event",
    "side", "entry", "stop", "target", "quantity_btc",
    "exit_price", "fx_rate",
    "risk_inr", "gross_pnl_inr", "fees_inr", "net_pnl_inr",
    "balance_inr", "equity_inr", "reason",
]



def append_journal(event, trade_id=None, pending_id=None, **fields):
    """Durable CSV append; initialize missing or empty journals safely."""
    rec = {
        "timestamp_utc": utc_now().isoformat(),
        "event_id": new_id("evt"),
        "trade_id": trade_id or "",
        "pending_id": pending_id or "",
        "event": event,
        **fields,
    }

    needs_header = (
        not os.path.exists(JOURNAL_FILE)
        or os.path.getsize(JOURNAL_FILE) == 0
    )

    with open(JOURNAL_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=JOURNAL_COLUMNS,
            extrasaction="ignore",
        )

        if needs_header:
            writer.writeheader()
            f.flush()
            os.fsync(f.fileno())

        writer.writerow(rec)
        f.flush()
        os.fsync(f.fileno())




def read_journal():
    """Returns (rows, errors). Empty-missing file -> ([], [])."""
    if not os.path.exists(JOURNAL_FILE):
        return [], []

    rows, errors = [], []

    try:
        with open(JOURNAL_FILE, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            if reader.fieldnames is None:
                return [], ["Journal has no header"]

            # Reject missing, duplicate, or unexpected header columns.
            header = reader.fieldnames
            if len(header) != len(set(header)):
                return [], ["Journal header contains duplicate columns"]

            missing = set(JOURNAL_COLUMNS) - set(header)
            unexpected = set(header) - set(JOURNAL_COLUMNS)

            if missing or unexpected:
                errors.append(
                    f"Invalid journal header: missing={sorted(missing)}, "
                    f"unexpected={sorted(unexpected)}"
                )
                return [], errors

            for i, row in enumerate(reader, start=2):
                if not row or not any(v is not None and v != "" for v in row.values()):
                    continue

                # Extra CSV values produce a None key in DictReader.
                if None in row:
                    errors.append(f"row {i}: extra CSV fields")
                    continue

                if any(row.get(col) is None for col in JOURNAL_COLUMNS):
                    errors.append(f"row {i}: missing CSV field values")
                    continue

                rows.append(row)

    except (OSError, UnicodeError, csv.Error) as exc:
        return [], [f"Journal read failed: {exc}"]

    return rows, errors



def index_journal(rows):
    """Return dicts keyed by trade_id / pending_id + duplicate/conflict ids."""
    by_tid, by_pid = {}, {}
    duplicate_event_ids, conflict_event_ids = set(), set()
    seen_event_ids = set()
    for r in rows:
        eid = r["event_id"]
        if eid in seen_event_ids:
            duplicate_event_ids.add(eid)
        seen_event_ids.add(eid)
        tid = r["trade_id"] or None
        pid = r["pending_id"] or None
        kind = r["event"]
        if tid:
            by_tid.setdefault(tid, []).append(r)
        if pid:
            by_pid.setdefault(pid, []).append(r)
    return by_tid, by_pid, duplicate_event_ids


# ============================================================ STATE

def initial_state():
    return {
        "version": SCHEMA_VERSION,
        "balance_inr": STARTING_BALANCE_INR,
        "day": utc_day(),
        "day_start_equity_inr": STARTING_BALANCE_INR,
        "trades_today": 0,
        "daily_realized_pnl_inr": 0.0,
        "consecutive_losses": 0,
        "halted_today": False,
        "pending": None,
        "position": None,
        "last_signal_bar": None,
        "last_processed_bar": None,
        "last_fx_rate": None,
        "last_fx_fetch_time": 0.0,
        "needs_recovery": False,
        "recovery_reason": "",
        "halt_reason": "",
    }


REQUIRED_KEYS = set(initial_state())


def _check_positive(p, keys, name):
    for k in keys:
        if k not in p:
            raise StateError(f"{name}.{k} missing")
        if not _finite(p[k]) or p[k] <= 0:
            raise StateError(f"{name}.{k} not positive finite: {p[k]!r}")


def _validate_pending(p):
    if p is None:
        return
    req = {"pending_id", "side", "entry", "stop", "target", "quantity_btc",
           "estimated_risk_inr", "estimated_net_rr", "bars_waited",
           "created_at", "signal_bar"}
    missing = req - set(p)
    if missing:
        raise StateError(f"pending missing: {sorted(missing)}")
    if p["side"] not in ("LONG", "SHORT"):
        raise StateError(f"pending.side: {p['side']!r}")
    _check_positive(p, ("entry", "stop", "target", "quantity_btc",
                        "estimated_risk_inr"), "pending")
    if p["side"] == "LONG" and not (p["stop"] < p["entry"] < p["target"]):
        raise StateError("LONG pending geometry violated")
    if p["side"] == "SHORT" and not (p["target"] < p["entry"] < p["stop"]):
        raise StateError("SHORT pending geometry violated")
    if p["bars_waited"] < 0 or not isinstance(p["bars_waited"], int):
        raise StateError("pending.bars_waited invalid")
    if p["estimated_risk_inr"] > RISK_CAP_INR + 1e-6:
        raise StateError(
            f"pending risk {p['estimated_risk_inr']:.2f} > cap")


def _validate_position(p):
    if p is None:
        return
    req = {"trade_id", "side", "entry_price", "stop", "target",
           "quantity_btc", "estimated_risk_inr", "entry_fee_inr",
           "entry_time", "fill_bar", "last_checked_bar"}
    missing = req - set(p)
    if missing:
        raise StateError(f"position missing: {sorted(missing)}")
    if not p["trade_id"]:
        raise StateError("position.trade_id empty")
    if p["side"] not in ("LONG", "SHORT"):
        raise StateError(f"position.side: {p['side']!r}")
    _check_positive(p, ("entry_price", "stop", "target", "quantity_btc",
                        "estimated_risk_inr", "entry_fee_inr"), "position")
    if p["side"] == "LONG" and not (p["stop"] < p["entry_price"] < p["target"]):
        raise StateError("LONG position geometry violated")
    if p["side"] == "SHORT" and not (p["target"] < p["entry_price"] < p["stop"]):
        raise StateError("SHORT position geometry violated")
    if p["fill_bar"] > p["last_checked_bar"]:
        raise StateError("position.last_checked_bar < fill_bar")
    if p["estimated_risk_inr"] > RISK_CAP_INR + 1e-6:
        raise StateError(
            f"position risk {p['estimated_risk_inr']:.2f} > cap "
            f"(stop widened after entry?)")


def validate_state(s):
    if not isinstance(s, dict):
        raise StateError("State not a JSON object")
    missing = REQUIRED_KEYS - set(s)
    if missing:
        raise StateError(f"State missing: {sorted(missing)}")
    if s["version"] != SCHEMA_VERSION:
        raise StateError(f"State version {s['version']} != {SCHEMA_VERSION}")

    if not _finite(s["balance_inr"]) or s["balance_inr"] < 0:
        raise StateError("balance_inr invalid")
    for k in ("day_start_equity_inr", "daily_realized_pnl_inr",
              "last_fx_fetch_time"):
        if not _finite(s[k]):
            raise StateError(f"{k} not finite")
    if s["last_fx_rate"] is not None and not (
            _finite(s["last_fx_rate"]) and FX_MIN < s["last_fx_rate"] < FX_MAX):
        raise StateError("last_fx_rate implausible")

    for k in ("trades_today", "consecutive_losses"):
        if not isinstance(s[k], int) or s[k] < 0:
            raise StateError(f"{k} invalid")
    if s["trades_today"] > MAX_TRADES_PER_DAY + 5:
        raise StateError(f"trades_today implausible: {s['trades_today']}")
    if not isinstance(s["halted_today"], bool):
        raise StateError("halted_today not bool")
    if not isinstance(s["needs_recovery"], bool):
        raise StateError("needs_recovery not bool")

    try:
        datetime.fromisoformat(s["day"])
    except Exception:
        raise StateError(f"day not ISO: {s['day']!r}")

    _validate_pending(s["pending"])
    _validate_position(s["position"])
    if s["pending"] is not None and s["position"] is not None:
        raise StateError("Both pending and position present")
    if s["halted_today"] and s["pending"] is not None:
        raise StateError("halted with pending order present")


def _backup_corrupt_state(reason):
    if not os.path.exists(STATE_FILE):
        return
    backup = f"{STATE_FILE}.corrupt.{int(time.time())}"
    try:
        shutil.copy2(STATE_FILE, backup)
        log(f"Backed up corrupt state ({reason}) -> {backup}")
    except Exception as exc:
        log(f"WARN: could not back up corrupt state: {exc}")


def load_state():
    if not os.path.exists(STATE_FILE):
        log("No state file; fresh account.")
        return initial_state()
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            s = json.load(f)
    except json.JSONDecodeError as exc:
        _backup_corrupt_state(f"JSON: {exc}")
        raise StateError(f"State corrupt ({exc}); backup saved; "
                         f"manual review required.")
    try:
        validate_state(s)
    except StateError:
        _backup_corrupt_state("validation")
        raise
    return s


def save_state(state):
    """Atomic write: tmp + fsync(file) + rename + fsync(parent dir)."""
    validate_state(state)
    tmp = f"{STATE_FILE}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, STATE_FILE)
        _fsync_dir(os.path.dirname(os.path.abspath(STATE_FILE)) or ".")
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass
        raise


def _fsync_dir(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        # Not supported on all platforms (e.g. Windows); non-fatal.
        pass


# ============================================================ STARTUP RECONCILE

def reconcile_state_with_journal(state, journal_rows):
    """
    Returns (repaired_state, needs_recovery, reason).
    Trusts journal for completed financial transitions.

    Repair rules (auto):
      * Journal has ENTRY_FILLED for pending_id P, and state has pending P
        (but no position): crash between journal and state save -> apply
        the fill from the journal.
      * Journal has POSITION_CLOSED for trade_id T, and state has position T:
        crash between journal and state save -> apply close from journal.
      * State has position T, no ENTRY_FILLED for T in journal: contradiction.
      * State has position T, journal has POSITION_CLOSED for T: contradiction.
      * Journal has duplicate event_id: contradiction.
    """
    by_tid, by_pid, dup_eids = index_journal(journal_rows)
    if dup_eids:
        return state, True, f"Journal has duplicate event_ids: {sorted(dup_eids)[:3]}"

    # Index latest close per trade.
    closes = {tid: [r for r in rows if r["event"] == "POSITION_CLOSED"]
              for tid, rows in by_tid.items()}

    needs_recovery = False
    reason = ""

    pos = state["position"]
    if pos:
        tid = pos["trade_id"]
        events = by_tid.get(tid, [])
        has_entry = any(e["event"] == "ENTRY_FILLED" for e in events)
        has_close = any(e["event"] == "POSITION_CLOSED" for e in events)
        if not has_entry:
            needs_recovery = True
            reason = f"Position {tid} has no ENTRY_FILLED in journal"
        elif has_close:
            needs_recovery = True
            reason = f"Position {tid} has POSITION_CLOSED in journal"

    if not pos:
        # Journal shows an ENTRY_FILLED, state has pending with same pending_id
        # => crash between journal and state.
        pend = state["pending"]
        if pend:
            pid = pend["pending_id"]
            for r in by_pid.get(pid, []):
                if r["event"] == "ENTRY_FILLED":
                    log(f"RECONCILE: replaying ENTRY_FILLED for pending {pid}")
                    tid = r["trade_id"]
                    # Rebuild position from journal + pending
                    fx = float(r["fx_rate"])
                    state["position"] = {
                        **pend,
                        "trade_id": tid,
                        "entry_price": float(r["entry"]),
                        "entry_fee_inr": float(r["fees_inr"]),
                        "entry_time": r["timestamp_utc"],
                        "fill_bar": int(r["fill_bar"]) if "fill_bar" in r else pend["signal_bar"],
                        "last_checked_bar": int(r["fill_bar"]) if "fill_bar" in r else pend["signal_bar"],
                    }
                    state["pending"] = None
                    state["trades_today"] += 1
                    save_state(state)
                    pos = state["position"]
                    break

    # Journal shows a close for a trade_id that state still has open.
    if state["position"]:
        tid = state["position"]["trade_id"]
        for r in closes.get(tid, []):
            log(f"RECONCILE: replaying POSITION_CLOSED for {tid}")
            # Recompute accounting deterministically from journal params.
            
            # Restore the exact P&L recorded by the original close event.
            net_pnl = _journal_net_pnl(r)


            state["balance_inr"] += net_pnl
            state["daily_realized_pnl_inr"] += net_pnl
            state["consecutive_losses"] = (state["consecutive_losses"] + 1
                                           if net_pnl < 0 else 0)
            state["position"] = None
            save_state(state)
            log(f"RECONCILE: applied close {tid} net=INR{net_pnl:.2f}")
            break

    if needs_recovery:
        state["needs_recovery"] = True
        state["recovery_reason"] = reason
        state["halted_today"] = True
        save_state(state)
    return state, needs_recovery, reason


# ============================================================ STRATEGY

def make_signal(bars5, bars15):
    if len(bars5) < 60 or len(bars15) < 60:
        return None, "Insufficient history"

    c5 = [b["close"] for b in bars5]
    c15 = [b["close"] for b in bars15]
    e9 = ema_series(c5, 9)
    e21 = ema_series(c5, 21)
    e50_15 = ema_series(c15, 50)
    if not e9 or not e21 or len(e50_15) < 6:
        return None, "Indicators not ready"

    last, prev = bars5[-1], bars5[-2]
    recent = bars5[-4:-1]
    a = atr(bars5, 14)
    if a is None or a <= 0:
        return None, "ATR unavailable"
    atr_pct = a / last["close"] * 100
    if not 0.10 <= atr_pct <= 1.20:
        return None, f"ATR% {atr_pct:.3f} out of range"
    avg_v = sum(b["volume"] for b in bars5[-21:-1]) / 20
    if avg_v <= 0 or last["volume"] < 0.80 * avg_v:
        return None, "Volume below threshold"

    trend = c15[-1] - e50_15[-1]
    slope = e50_15[-1] - e50_15[-5]
    bull = c15[-1] > e50_15[-1] and slope > 0
    bear = c15[-1] < e50_15[-1] and slope < 0
    zl, zh = min(e9[-3:]), max(e21[-3:])
    long_pb = any(b["low"] <= max(e9[-1], e21[-1])
                  and b["low"] >= zl - a for b in recent)
    short_pb = any(b["high"] >= min(e9[-1], e21[-1])
                   and b["high"] <= zh + a for b in recent)

    long_ok = (bull and e9[-1] > e21[-1] and last["close"] > last["open"]
               and last["close"] > e9[-1] and last["close"] > prev["high"]
               and long_pb)
    short_ok = (bear and e9[-1] < e21[-1] and last["close"] < last["open"]
                and last["close"] < e9[-1] and last["close"] < prev["low"]
                and short_pb)
    if not long_ok and not short_ok:
        return None, "No setup"

    side = "LONG" if long_ok else "SHORT"
    swing = bars5[-5:]
    if side == "LONG":
        stop = min(b["low"] for b in swing) - 0.15 * a
        entry = last["close"]
        dist = entry - stop
    else:
        stop = max(b["high"] for b in swing) + 0.15 * a
        entry = last["close"]
        dist = stop - entry
    if dist <= 0:
        return None, "Invalid stop distance"
    dist_atr = dist / a
    if not 0.80 <= dist_atr <= 2.50:
        return None, f"Stop distance {dist_atr:.2f} ATR out of range"
    tgt = entry + TARGET_R_MULTIPLE * dist if side == "LONG" \
          else entry - TARGET_R_MULTIPLE * dist

    return {
        "side": side, "entry": entry, "stop": stop, "target": tgt,
        "atr": a, "signal_bar": last["open_time"],
        "reason": f"{side}: 15m trend + 5m pullback; ATR%={atr_pct:.3f} "
                  f"volratio={last['volume']/avg_v:.2f}",
    }, "Candidate"


# ============================================================ SIZING

def round_trip_rate():
    return 2 * (FEE_PER_SIDE + SLIPPAGE_PER_SIDE)


def size_position(sig, state, fx):
    entry, stop = sig["entry"], sig["stop"]
    dist = abs(entry - stop)
    rt = round_trip_rate()
    risk_per_btc_inr = (dist + entry * rt) * fx
    if risk_per_btc_inr <= 0:
        return None, "Non-positive risk per BTC"
    risk_budget = min(RISK_CAP_INR, state["balance_inr"] * 0.005)
    qty_risk = risk_budget / risk_per_btc_inr
    qty_notional = state["balance_inr"] / (entry * fx)
    qty = min(qty_risk, qty_notional)
    if qty <= 0:
        return None, "Non-positive quantity"
    est_risk = qty * risk_per_btc_inr
    if est_risk > RISK_CAP_INR + 1e-6:
        return None, f"Risk {est_risk:.2f} > cap"
    net_reward = abs(sig["target"] - entry) - entry * rt
    net_risk = dist + entry * rt
    if net_reward <= 0 or net_risk <= 0:
        return None, "Net costs exceed reward"
    net_rr = net_reward / net_risk
    if net_rr < MIN_NET_RR:
        return None, f"Net R:R {net_rr:.2f} < {MIN_NET_RR:.2f}"
    return {"quantity_btc": qty, "estimated_risk_inr": est_risk,
            "estimated_net_rr": net_rr}, ""


# ============================================================ ACCOUNTING

def exit_slip_price(mark, side):
    return mark * (1 - SLIPPAGE_PER_SIDE) if side == "LONG" \
           else mark * (1 + SLIPPAGE_PER_SIDE)


def unrealized_inr(state, mark, fx):
    pos = state["position"]
    if not pos:
        return 0.0
    d = 1 if pos["side"] == "LONG" else -1
    sp = exit_slip_price(mark, pos["side"])
    gross = (sp - pos["entry_price"]) * pos["quantity_btc"] * d
    fees = pos["entry_fee_inr"] + pos["quantity_btc"] * sp * fx * FEE_PER_SIDE
    return gross * fx - fees


def equity_inr(state, mark, fx):
    return state["balance_inr"] + unrealized_inr(state, mark, fx)


# ============================================================ PENDING

def set_pending(sig, sz, state):
    pid = new_id("ord")
    state["pending"] = {**sig, **sz, "pending_id": pid,
                        "created_at": utc_now().isoformat(),
                        "bars_waited": 0}
    append_journal(
        "LIMIT_ORDER_PLACED", pending_id=pid,
        side=sig["side"], entry=sig["entry"], stop=sig["stop"],
        target=sig["target"], quantity_btc=sz["quantity_btc"],
        risk_inr=sz["estimated_risk_inr"], fx_rate=state["last_fx_rate"],
        reason=sig["reason"], balance_inr=state["balance_inr"],
        equity_inr=state["balance_inr"],
    )
    save_state(state)
    log(f"LIMIT {sig['side']} {sig['entry']:.2f} stop={sig['stop']:.2f} "
        f"tgt={sig['target']:.2f} qty={sz['quantity_btc']:.6f} "
        f"risk=INR{sz['estimated_risk_inr']:.2f} R:R={sz['estimated_net_rr']:.2f}")


def cancel_pending(state, reason):
    p = state["pending"]
    if not p:
        return
    append_journal(
        "LIMIT_ORDER_CANCELLED", pending_id=p["pending_id"],
        side=p["side"], entry=p["entry"], stop=p["stop"],
        target=p["target"], quantity_btc=p["quantity_btc"],
        risk_inr=p["estimated_risk_inr"], reason=reason,
        fx_rate=state["last_fx_rate"],
        balance_inr=state["balance_inr"], equity_inr=state["balance_inr"],
    )
    state["pending"] = None
    save_state(state)


# ============================================================ FILL / CLOSE

def fill_pending(state, bar, fx):
    """Limit-touch => fill at entry (assumption; see header)."""
    p = state["pending"]
    if not p:
        return False
    entry = p["entry"]
    touched = (bar["low"] <= entry if p["side"] == "LONG"
               else bar["high"] >= entry)
    if not touched:
        return False

    pid = p["pending_id"]
    tid = new_id("trade")
    entry_fee = p["quantity_btc"] * entry * fx * FEE_PER_SIDE

    # Journal FIRST — idempotency anchor.
    append_journal(
        "ENTRY_FILLED", trade_id=tid, pending_id=pid,
        side=p["side"], entry=entry, stop=p["stop"], target=p["target"],
        quantity_btc=p["quantity_btc"], risk_inr=p["estimated_risk_inr"],
        fees_inr=entry_fee, fx_rate=fx, fill_bar=bar["open_time"],
        reason=p["reason"], balance_inr=state["balance_inr"],
        equity_inr=state["balance_inr"],
    )
    # Then state.
    state["position"] = {
        **p, "trade_id": tid, "entry_price": entry,
        "entry_fee_inr": entry_fee,
        "entry_time": utc_now().isoformat(),
        "fill_bar": bar["open_time"],
        "last_checked_bar": bar["open_time"],
    }
    state["pending"] = None
    state["trades_today"] += 1
    save_state(state)
    log(f"FILL trade={tid} {p['side']} @ {entry:.2f}")
    return True


def close_position(state, fx, ref_price, reason):
    """Journal FIRST, then state. Idempotent: if no position, no-op."""
    pos = state["position"]
    if not pos:
        return

    tid = pos["trade_id"]
    side, entry, qty = pos["side"], pos["entry_price"], pos["quantity_btc"]
    slip = (ref_price * (1 - SLIPPAGE_PER_SIDE) if side == "LONG"
            else ref_price * (1 + SLIPPAGE_PER_SIDE))
    gross = ((slip - entry) * qty if side == "LONG"
             else (entry - slip) * qty)
    exit_fee = qty * slip * fx * FEE_PER_SIDE
    gross_inr = gross * fx
    total_fees = pos["entry_fee_inr"] + exit_fee
    net_pnl = gross_inr - total_fees

    # Journal FIRST.
    append_journal(
        "POSITION_CLOSED", trade_id=tid,
        side=side, entry=entry, stop=pos["stop"], target=pos["target"],
        quantity_btc=qty, exit_price=ref_price, fx_rate=fx,
        risk_inr=pos["estimated_risk_inr"],
        gross_pnl_inr=gross_inr, fees_inr=total_fees, net_pnl_inr=net_pnl,
        balance_inr=state["balance_inr"] + net_pnl,
        equity_inr=state["balance_inr"] + net_pnl,
        reason=reason,
    )
    # Then state.
    state["balance_inr"] += net_pnl
    state["daily_realized_pnl_inr"] += net_pnl
    if net_pnl < 0:
        state["consecutive_losses"] += 1
    else:
        state["consecutive_losses"] = 0
    state["position"] = None
    save_state(state)
    log(f"EXIT trade={tid} {reason} @ {ref_price:.2f} net=INR{net_pnl:.2f} "
        f"bal=INR{state['balance_inr']:.2f} "
        f"consec={state['consecutive_losses']}")


# ============================================================ CANDLE PROCESS

def bar_exit_check(state, bar, fx):
    pos = state["position"]
    if not pos:
        return False
    is_fill = (bar["open_time"] == pos["fill_bar"])
    stop, target = pos["stop"], pos["target"]
    if pos["side"] == "LONG":
        stop_hit = bar["low"] <= stop
        target_hit = (not is_fill) and bar["high"] >= target
        stop_ref = stop if is_fill else min(stop, bar["open"])
    else:
        stop_hit = bar["high"] >= stop
        target_hit = (not is_fill) and bar["low"] <= target
        stop_ref = stop if is_fill else max(stop, bar["open"])
    if stop_hit:
        close_position(state, fx, stop_ref, "STOP_LOSS")
        return True
    if target_hit:
        close_position(state, fx, target, "TAKE_PROFIT")
        return True
    if not is_fill:
        state["position"]["last_checked_bar"] = bar["open_time"]
        save_state(state)
    return False


def check_pending_on_bar(state, bar, fx):
    """Process pending on a single candle. Fills, expiry, or no-op."""
    p = state["pending"]
    if not p:
        return
    p["bars_waited"] += 1
    save_state(state)
    if fill_pending(state, bar, fx):
        # Fill candle: stop wins, target NOT credited.
        bar_exit_check(state, bar, fx)
        return
    if p["bars_waited"] >= PENDING_EXPIRY_BARS:
        cancel_pending(state, "Expired after 3 bars")


def process_new_candles(state, bars5, fx):
    """Chronological. Checkpoint advances only after each candle is durable."""
    if state["last_processed_bar"] is None:
        # Fresh state: adopt watermark without back-processing.
        state["last_processed_bar"] = bars5[-1]["open_time"]
        save_state(state)
        return

    for bar in bars5:
        if bar["open_time"] <= state["last_processed_bar"]:
            continue
        # 1) Pending fill/expiry (may create position, exit fills candle).
        if state["pending"]:
            check_pending_on_bar(state, bar, fx)
        # 2) Position stop/target on this candle.
        if state["position"]:
            if bar["open_time"] > state["position"]["last_checked_bar"]:
                bar_exit_check(state, bar, fx)
        # 3) Advance checkpoint.
        state["last_processed_bar"] = bar["open_time"]
        save_state(state)


def monitor_position_live(state, mark, fx):
    pos = state["position"]
    if not pos:
        return
    side, stop, target = pos["side"], pos["stop"], pos["target"]
    if side == "LONG":
        if mark <= stop:
            close_position(state, fx, min(stop, mark), "STOP_LOSS")
        elif mark >= target:
            close_position(state, fx, target, "TAKE_PROFIT")
    else:
        if mark >= stop:
            close_position(state, fx, max(stop, mark), "STOP_LOSS")
        elif mark <= target:
            close_position(state, fx, target, "TAKE_PROFIT")


# ============================================================ BREAKERS

def enforce_daily_breaker(state, mark, fx):
    if state["halted_today"]:
        return True
    eq = equity_inr(state, mark, fx)
    dd = state["day_start_equity_inr"] - eq
    if dd < DAILY_LOSS_LIMIT_INR:
        return False
    log(f"DAILY LOSS BREAKER dd=INR{dd:.2f} >= {DAILY_LOSS_LIMIT_INR:.2f}")
    if state["pending"]:
        cancel_pending(state, "Daily loss breaker")
    if state["position"]:
        close_position(state, fx, mark, "DAILY_LOSS_BREAKER")
    state["halted_today"] = True
    state["halt_reason"] = "DAILY_LOSS"
    save_state(state)
    return True


def enforce_consecutive_loss_limit(state):
    if state["consecutive_losses"] < MAX_CONSECUTIVE_LOSSES:
        return False
    if state["halted_today"]:
        return True
    log(f"CONSECUTIVE LOSS HALT ({state['consecutive_losses']})")
    if state["pending"]:
        cancel_pending(state, "Consecutive-loss halt")
    state["halted_today"] = True
    state["halt_reason"] = "CONSECUTIVE_LOSSES"
    save_state(state)
    return True


# ============================================================ ROLLOVER

def reconcile_and_rollover(state, bars5, fx):
    """Reconcile old candles against old baseline, then establish new day."""
    # 1. Replay unseen candles under the OLD day's state.
    process_new_candles(state, bars5, fx)

    today = utc_day()
    if state["day"] != today:
        # 2. Establish new baseline from RECONCILED equity.
        try:
            mark = get_mark_price()
        except Exception:
            mark = None
        if mark is None:
            # Cannot establish meaningful baseline; defer to next cycle.
            return
        new_eq = equity_inr(state, mark, fx)
        state["day"] = today
        state["day_start_equity_inr"] = new_eq
        state["trades_today"] = 0
        state["daily_realized_pnl_inr"] = 0.0
        # Reset daily halt ONLY if it was for DAILY_LOSS.
        if state.get("halt_reason") == "DAILY_LOSS":
            state["halted_today"] = False
            state["halt_reason"] = ""
        # Consecutive-loss halt and needs_recovery PERSIST.
        save_state(state)
        log(f"ROLLOVER day={today} baseline=INR{new_eq:.2f} "
            f"consec_kept={state['consecutive_losses']}")


# ============================================================ MAIN CYCLE

def run_cycle(state):
    bars5 = get_klines("5m")
    bars15 = get_klines("15m")

    fx, fx_fresh = refresh_fx(state)
    mark = get_mark_price()

    reconcile_and_rollover(state, bars5, fx)

    if state["position"]:
        monitor_position_live(state, mark, fx)

    enforce_daily_breaker(state, mark, fx)
    enforce_consecutive_loss_limit(state)

    if state["needs_recovery"]:
        return POLL_SECONDS
    if state["halted_today"]:
        return POLL_SECONDS
    if state["pending"] or state["position"]:
        return POLL_SECONDS
    if state["trades_today"] >= MAX_TRADES_PER_DAY:
        return POLL_SECONDS
    if state["balance_inr"] <= 0:
        log("Balance depleted; stopping.")
        return None
    if not fx_fresh and fx is None:
        return POLL_SECONDS

    # Only generate a new signal on a bar newer than last_processed? The
    # watermark advances in process_new_candles; if it advanced this cycle,
    # we may still want to signal on that bar (strategies use bar[-1]).
    sig, reason = make_signal(bars5, bars15)
    if sig is None:
        log(f"No signal: {reason}")
        return POLL_SECONDS
    if state["last_signal_bar"] == sig["signal_bar"]:
        return POLL_SECONDS
    state["last_signal_bar"] = sig["signal_bar"]

    sz, why = size_position(sig, state, fx)
    if sz is None:
        append_journal(
            "SIGNAL_REJECTED",
            side=sig["side"], entry=sig["entry"], stop=sig["stop"],
            target=sig["target"],
            reason=f"Sizing/netRR failed: {why}",
            fx_rate=fx,
            balance_inr=state["balance_inr"],
            equity_inr=equity_inr(state, mark, fx),
        )
        log(f"Rejected: {why}")
        save_state(state)
    else:
        set_pending(sig, sz, state)
    return POLL_SECONDS


def safe_cycle(state):
    try:
        return run_cycle(state)
    except RateLimited as exc:
        log(f"RATE LIMITED: {exc.retry_after}s")
        return exc.retry_after
    except DataError as exc:
        log(f"DATA ERROR: {exc}")
        state["needs_recovery"] = True
        state["recovery_reason"] = f"Data: {exc}"
        state["halted_today"] = True
        try:
            save_state(state)
        except Exception:
            logging.exception("Could not persist recovery flag")
        return 30
    except requests.RequestException as exc:
        log(f"HTTP ERROR: {exc}. Retrying.")
        return 15
    except StateError as exc:
        log(f"STATE ERROR: {exc}. Stopping.")
        return None
    except Exception as exc:
        logging.exception("Unexpected engine error")
        log(f"UNEXPECTED: {exc}. Retrying.")
        try:
            save_state(state)
        except Exception:
            logging.exception("State save failed")
        return 20


# ============================================================ MAIN

def _assert_paper_only():
    if not PAPER_ONLY:
        raise SystemExit("PAPER_ONLY=False — refusing to run.")



def main():
    _assert_paper_only()

    lock = InstanceLock(LOCK_FILE)
    if not lock.acquire():
        log("Another engine instance is running (lock held). Exiting.")
        return

    try:
        try:
            state = load_state()
        except StateError as exc:
            log(f"FATAL: {exc}")
            return

        journal_rows, journal_errors = read_journal()

        if journal_errors:
            log(
                "FATAL: Journal validation failed; "
                "startup stopped without reconciliation. "
                f"Errors: {journal_errors[:3]}"
            )
            return

        state, flagged, why = reconcile_state_with_journal(
            state, journal_rows
        )
        if flagged:
            log(f"STARTUP: needs_recovery set: {why}")

        log("=" * 60)
        log("BTCUSDT PAPER ENGINE v3.0 — SIMULATION ONLY")
        log(
            f"balance=INR{state['balance_inr']:.2f} "
            f"baseline=INR{state['day_start_equity_inr']:.2f} "
            f"trades_today={state['trades_today']} "
            f"consec={state['consecutive_losses']} "
            f"halted={state['halted_today']} "
            f"needs_recovery={state['needs_recovery']}"
        )
        if state["needs_recovery"]:
            log(f"RECOVERY REASON: {state['recovery_reason']}")
        log("=" * 60)

        while True:
            try:
                sleep_s = safe_cycle(state)
            except KeyboardInterrupt:
                try:
                    save_state(state)
                    log("Stopped by user. State saved.")
                except Exception:
                    logging.exception("Shutdown save failed")
                break

            if sleep_s is None:
                break

            try:
                time.sleep(sleep_s)
            except KeyboardInterrupt:
                try:
                    save_state(state)
                    log("Stopped by user. State saved.")
                except Exception:
                    logging.exception("Shutdown save failed")
                break

    finally:
        lock.release()
