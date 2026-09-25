from __future__ import annotations

import pytest

from ocr_pipeline.common import _TABLE_SCALAR, _numeric_value
from ocr_pipeline.table_blocks import _reinterpret_period_thousands, _table_total_reconciliation


@pytest.mark.parametrize(("text", "expected"), [
    ("-$21,903", -21903), ("$-21,903", -21903), ("-26,512", -26512),
    ("($1,250)", -1250), ("(45,670)", -45670), ("(3.5%)", -3.5),
    ("−5.0%", -5.0), ("–0.8 to –1.5", -0.8),
    ("$9.183.414", 9183414), ("1.234,56", 1234.56), ("$4.437M", 4.437), ("$1,835.75", 1835.75),
])
def test_signs_and_separators_are_read_from_the_printed_number(text: str, expected: float) -> None:
    assert _numeric_value(text)[0] == pytest.approx(expected)


def test_spaced_dashes_and_footnote_markers_are_not_signs() -> None:
    assert _numeric_value("$450 - $550")[0] == 450
    assert _numeric_value("Townhouses – 100%")[0] == 100
    assert _numeric_value("(1)") == (None, None, None)
    assert _numeric_value("Closing Costs (1) 387,810")[0] == 387810
    assert _numeric_value("(1,173 sqft")[0] == 1173


def test_signed_and_parenthesized_amounts_are_table_scalars() -> None:
    for text in ["-$21,903", "$-21,903", "($1,250)", "(3.5%)", "−5.0%", "$9.183.414"]:
        assert _TABLE_SCALAR.fullmatch(text), text
    assert not _TABLE_SCALAR.fullmatch("$1,250 $2,500")


def _row(label: str, *values: str) -> dict:
    cells = []
    for index, raw in enumerate(values, 2):
        numeric, unit, normalized = _numeric_value(raw)
        cells.append({"column_id": f"c{index:03d}", "raw_value": raw, "value_state": "present",
                      "numeric_value": numeric, "unit": unit, "normalized_value": normalized})
    return {"label": label, "cells": cells}


def test_period_thousands_follow_comma_peers_only() -> None:
    rows = [_row("Total Potential Income", "$438,056", "$442,685", "$458.056", "$472,983"),
            _row("Vacancy", "-$21,903", "-$22,134", "-$22,903", "-$23.649")]
    assert _reinterpret_period_thousands(rows) == 2
    assert rows[0]["cells"][2]["numeric_value"] == 458056
    assert rows[1]["cells"][3]["numeric_value"] == -23649

    prices = [_row("Unit price", "$1.125", "$1.250", "$1.375")]
    assert _reinterpret_period_thousands(prices) == 0
    assert prices[0]["cells"][0]["numeric_value"] == pytest.approx(1.125)


def test_subtotals_reconcile_with_signed_values() -> None:
    rows = [["Line", "Year 1"], ["Total Revenue", "$1,000"], ["Total Expenses", "-$400"],
            ["Other", "$5"], ["Total Net", "$600"]]
    result = _table_total_reconciliation(rows)
    assert result and result["passed"]
