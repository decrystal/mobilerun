"""Portal's timeout cache is not evidence of the current screen."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from mobilerun.tools.ui.provider import fetch_state_with_retry


def state(*, degraded=False):
    return {
        "a11y_tree": {"text": "Copy", "children": []},
        "phone_state": {"packageName": "com.android.settings"},
        "device_context": {},
        **(
            {"degraded": True, "degradedReason": "a11y_timeout", "snapshotAgeMs": 10000}
            if degraded
            else {}
        ),
    }


def test_cached_state_is_retried_before_publishing_current_controls():
    current = state()
    current["a11y_tree"]["text"] = "Current screen"
    fetch = AsyncMock(side_effect=[state(degraded=True), current])
    recovery = AsyncMock()
    result = asyncio.run(
        fetch_state_with_retry(fetch, recovery, max_retries=2, retry_delays=[0])
    )
    assert result is current
    assert fetch.await_count == 2
    recovery.assert_not_awaited()


def test_repeated_cached_state_exhausts_existing_retry_budget_without_returning_it():
    fetch = AsyncMock(return_value=state(degraded=True))
    recovery = AsyncMock()
    with pytest.raises(Exception, match="cached accessibility snapshot"):
        asyncio.run(
            fetch_state_with_retry(
                fetch, recovery, max_retries=3, retry_delays=[0], recovery_after=2
            )
        )
    assert fetch.await_count == 3
    recovery.assert_awaited_once()


def test_fresh_empty_tree_remains_valid_for_visual_observation():
    current = state()
    current.update(a11y_tree={}, degraded=False)
    fetch = AsyncMock(return_value=current)
    assert asyncio.run(fetch_state_with_retry(fetch, max_retries=1)) is current
