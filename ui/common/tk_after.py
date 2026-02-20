from __future__ import annotations

from typing import Callable, Any


def safe_after(*, root, after_ids: list[str], ms: int, fn: Callable[[], Any]) -> str:
    """Schedule `fn` via Tk `after` and remember the returned id.

    Pass in the same `after_ids` list everywhere so we can cancel pending callbacks on destroy.
    """
    aid = root.after(int(ms), fn)
    after_ids.append(aid)
    return aid


def make_on_destroy(*, root, after_ids: list[str]):
    """Return a <Destroy> handler that cancels scheduled `after` callbacks."""

    def _handler(event=None) -> None:
        # Only handle the root window destroy event
        try:
            if event is not None and event.widget is not root:
                return
        except Exception:
            pass

        for aid in list(after_ids):
            try:
                root.after_cancel(aid)
            except Exception:
                pass
        after_ids.clear()

    return _handler