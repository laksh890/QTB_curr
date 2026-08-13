"""Accounting: capital, positions, orders, fills, trades, reconciliation."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from iqrp.app.backtesting.accounting import (
    CapitalState,
    FillLog,
    FillRecord,
    OrderLog,
    OrderRecord,
    PortfolioSnapshot,
    PositionBook,
    PositionRecord,
    ReconciliationError,
    SnapshotBook,
    TradeLedger,
    TradeRecord,
    reconcile_capital,
)


def test_capital_state_lifecycle():
    cap = CapitalState(initial_capital=100_000.0)
    assert cap.cash == 100_000.0
    assert cap.equity == 100_000.0
    cap.apply_cash_delta(-10_000.0, reason="buy")
    cap.mark_unrealized(500.0, market_value=10_500.0)
    assert abs(cap.equity - (cap.cash + cap.position_market_value)) < 1e-9
    cap.record_fee(5.0)
    cap.record_financing(2.0)
    cap.realize(100.0, settle_into_cash=True)
    d = cap.to_dict()
    restored = CapitalState.from_dict(d)
    assert restored.cash == cap.cash
    assert restored.available_cash == restored.cash - restored.margin_used
    assert restored.free_cash >= 0
    assert restored.margin == restored.margin_used


def test_position_book_open_reduce_flip():
    book = PositionBook()
    realized = book.apply_fill("AAA", quantity=10, price=100.0, side="buy")
    assert realized == 0.0
    assert book.get("AAA").quantity == 10
    book.mark_all({"AAA": 110.0})
    assert book.total_unrealized() > 0
    realized2 = book.apply_fill("AAA", quantity=4, price=110.0, side="sell")
    assert realized2 > 0
    # Flip
    realized3 = book.apply_fill("AAA", quantity=20, price=105.0, side="sell")
    assert book.get("AAA").quantity < 0
    assert "AAA" in book.open_instruments()
    snap = book.snapshot()
    assert snap
    d = book.to_dict()
    book2 = PositionBook.from_dict(d)
    assert book2.get("AAA").quantity == book.get("AAA").quantity
    assert list(book.iter_open())
    assert book.quantities()
    assert book.market_values()
    assert book.total_exposure() >= 0
    assert book.total_realized() != 0 or realized3 is not None
    # Cover to flat
    book.apply_fill("AAA", quantity=abs(book.get("AAA").quantity), price=100.0, side="buy")
    # Add then reduce to zero
    book.apply_fill("BBB", quantity=5, price=50.0, side="buy")
    book.apply_fill("BBB", quantity=5, price=55.0, side="sell")
    assert abs(book.get("BBB").quantity) < 1e-12
    rec = PositionRecord.from_dict(PositionRecord("CCC", quantity=1, price=1).to_dict())
    assert rec.instrument == "CCC"


def test_orders_fills_trades_logs():
    orders = OrderLog()
    o = orders.add(
        OrderRecord(
            order_id="o1",
            timestamp=OrderLog.ts_str(datetime(2020, 1, 2, tzinfo=UTC)),
            instrument="AAA",
            side="buy",
            quantity=10,
        )
    )
    assert len(orders) == 1
    orders.add({"order_id": "o2", "timestamp": "t", "instrument": "BBB", "side": "sell", "quantity": 1})
    assert OrderLog.from_dict(orders.to_dict()).to_list()

    fills = FillLog()
    fills.add(
        FillRecord(
            fill_id="f1",
            order_id="o1",
            timestamp="t",
            instrument="AAA",
            side="buy",
            quantity=10,
            price=100.0,
            fee=1.0,
        )
    )
    fills.add({"fill_id": "f2", "order_id": "o2", "timestamp": "t", "instrument": "BBB", "side": "sell", "quantity": 1, "price": 50})
    assert len(fills) == 2
    assert FillLog.from_dict(fills.to_dict()).to_list()

    ledger = TradeLedger()
    ledger.open_trade(
        trade_id="t1",
        instrument="AAA",
        side="long",
        quantity=10,
        entry_time="t0",
        entry_price=100.0,
    )
    closed = ledger.close_trade("AAA", exit_time="t1", exit_price=110.0, realized_pnl=100.0)
    assert closed is not None and closed.status == "closed"
    assert ledger.close_trade("MISSING", exit_time="t", exit_price=1.0, realized_pnl=0) is None
    ledger.record_fill_as_trade(
        trade_id="t2",
        instrument="BBB",
        side="buy",
        quantity=1,
        timestamp="t",
        price=50,
        realized_pnl=0.0,
    )
    ledger.record_fill_as_trade(
        trade_id="t3",
        instrument="BBB",
        side="sell",
        quantity=1,
        timestamp="t",
        price=55,
        realized_pnl=5.0,
    )
    assert TradeLedger.from_dict(ledger.to_dict()).to_list()
    tr = TradeRecord.from_dict(TradeRecord("x", "Y", "long", 1, "t", 1.0).to_dict())
    assert tr.trade_id == "x"


def test_snapshots_and_reconciliation_success_failure():
    book = SnapshotBook()
    book.add(
        PortfolioSnapshot(
            timestamp="2020-01-02T00:00:00+00:00",
            equity=100_000.0,
            cash=50_000.0,
            positions={"AAA": 10},
            gross_exposure=50_000.0,
            net_exposure=50_000.0,
            leverage=0.5,
        )
    )
    book.add(
        {
            "timestamp": "2020-01-03T00:00:00+00:00",
            "equity": 101_000.0,
            "cash": 51_000.0,
            "positions": {},
        }
    )
    curve = book.equity_curve()
    assert len(curve) >= 1
    assert SnapshotBook.from_dict(book.to_dict()).to_list()

    # Success via cash identity (realized settled into cash)
    cap = CapitalState(initial_capital=1000.0)
    cap.apply_cash_delta(-500.0)
    cap.mark_unrealized(20.0, market_value=520.0)
    cap.record_fee(1.0)
    ok = reconcile_capital(cap, fail=True)
    assert ok.ok
    assert ok.to_dict()["ok"] is True

    # Mapping form success
    ok2 = reconcile_capital(cap.to_dict(), fail=False)
    assert ok2.ok

    # Failure
    broken = {
        "initial_capital": 1000.0,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "fees_paid": 0.0,
        "financing_paid": 0.0,
        "equity": 500.0,
        "cash": 100.0,
    }
    with pytest.raises(ReconciliationError):
        reconcile_capital(broken, ending_equity=500.0, tolerance=1e-6, fail=True)
    bad = reconcile_capital(broken, ending_equity=500.0, tolerance=1e-6, fail=False)
    assert bad.ok is False
