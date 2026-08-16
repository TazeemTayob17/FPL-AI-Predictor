# Checks the "time ago" formatter used by the dashboard's staleness indicator.

from datetime import datetime, timedelta, timezone

from fpl_agent.ui.components.staleness import format_time_ago

NOW = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)


# Under a minute must read as "just now", not "0m ago".
def test_format_time_ago_under_a_minute_is_just_now():
    assert format_time_ago(NOW - timedelta(seconds=30), now=NOW) == "just now"


# Minutes-scale gaps must be reported in minutes.
def test_format_time_ago_reports_minutes():
    assert format_time_ago(NOW - timedelta(minutes=5), now=NOW) == "5m ago"


# Hours-scale gaps must be reported in hours, not a large minute count.
def test_format_time_ago_reports_hours():
    assert format_time_ago(NOW - timedelta(hours=3), now=NOW) == "3h ago"


# Day-scale gaps must be reported in days.
def test_format_time_ago_reports_days():
    assert format_time_ago(NOW - timedelta(days=2), now=NOW) == "2d ago"
