"""Ephemeral captions. Only played assistant text is exposed to the browser."""

import asyncio
from collections import OrderedDict


class TranscriptLedger:
    def __init__(self):
        self.rows = OrderedDict()
        self.revision = 0
        self.changed = asyncio.Event()
        self.closed = False
        self.items = OrderedDict()
        self.user_turn = None
        self.last_speech_end = None
        self.lookup_token = None
        self.lookup_state = "idle"

    def begin_lookup(self, token):
        if not self.closed:
            self.lookup_token, self.lookup_state = token, "checking"
            self._touch()

    def end_lookup(self, token, state="idle"):
        if (
            not self.closed
            and token == self.lookup_token
            and state != self.lookup_state
        ):
            self.lookup_state = state
            self._touch()

    def clear_lookup(self):
        self.end_lookup(self.lookup_token)
        self.lookup_token = None

    def sources(self, response_id, sources):
        from .web_search import safe_source

        row = self.rows.get("assistant:" + response_id)
        if row and not self.closed:
            cleaned = [
                source
                for item in sources[:5]
                if (source := safe_source(item.get("url"), item.get("title", "")))
            ]
            if cleaned:
                row["sources"] = cleaned
                self._touch()

    def speech_started(self, item_id, start_ms=None):
        # VAD may commit a hesitation as a separate provider item. Preserve one
        # bubble across a short pause until the assistant has actually spoken.
        if (
            start_ms is None
            or self.last_speech_end is None
            # Provider VAD includes prefix padding, so adjacent segments can
            # overlap by a few hundred milliseconds on the source-audio clock.
            or not -500 <= start_ms - self.last_speech_end <= 1500
        ):
            self.user_turn = None
        self._user_item(item_id, self.user_turn)
        self.user(item_id)

    def speech_stopped(self, item_id, end_ms=None):
        self.last_speech_end = end_ms

    def _user_item(self, item_id, group=None):
        if item_id not in self.items:
            if group not in self.rows:
                group = "user:" + item_id
            self._row(group, "user")
            self.items[item_id] = dict(group=group, text="", final=False)
            self.user_turn = group
            while len(self.items) > 600:
                self.items.popitem(last=False)
        return self.items[item_id]

    def _touch(self):
        self.revision += 1
        self.changed.set()

    def _row(self, ident, speaker, epoch=None):
        if ident not in self.rows:
            self.rows[ident] = dict(
                id=ident,
                speaker=speaker,
                text="",
                state="partial",
                epoch=epoch,
                cues=[],
                total=None,
                played=0,
                order=self.revision,
            )
            while len(self.rows) > 200:
                self.rows.popitem(last=False)
            self._touch()
        return self.rows[ident]

    def user(self, item_id, text=None, *, final=False, append=False):
        if self.closed:
            return
        item = self._user_item(item_id)
        row = self.rows.get(item["group"])
        if not row or item["final"]:
            return
        # A commit is a placeholder, not a replacement for earlier live deltas.
        if text is not None:
            item["text"] = item["text"] + text if append else text
        item["final"] = final
        parts = [part for part in self.items.values() if part["group"] == row["id"]]
        row["text"] = " ".join(
            part["text"].strip() for part in parts if part["text"].strip()
        )
        row["state"] = "complete" if all(part["final"] for part in parts) else "partial"
        self._touch()

    def begin(self, response_id, epoch):
        if not self.closed:
            self._row("assistant:" + response_id, "assistant", epoch)

    def cue(self, response_id, sample, cumulative_text):
        row = self.rows.get("assistant:" + response_id)
        if not row or row["state"] != "partial" or self.closed:
            return
        row["cues"].append((max(0, int(sample)), cumulative_text))
        self.advance(row["epoch"], row["played"])

    def finish(self, response_id, total):
        row = self.rows.get("assistant:" + response_id)
        if row and row["state"] == "partial":
            row["total"] = total
            self.advance(row["epoch"], row["played"])

    def playback_complete(self, response_id):
        row = self.rows.get("assistant:" + response_id)
        return bool(row and row["state"] == "complete")

    def advance(self, epoch, samples, *, complete=False):
        for row in self.rows.values():
            if (
                row["speaker"] != "assistant"
                or row["epoch"] != epoch
                or row["state"] != "partial"
            ):
                continue
            row["played"] = max(row["played"], samples)
            if complete and row["total"] is not None:
                row["played"] = max(row["played"], row["total"])
            text = row["text"]
            for position, value in row["cues"]:
                if position <= row["played"] and row["played"] > 0:
                    text = value
            state = (
                "complete"
                if row["total"] is not None
                and row["played"] >= row["total"]
                and (row["cues"] or row["total"] == 0)
                else "partial"
            )
            if text != row["text"] or state != row["state"]:
                if text and not row["text"]:
                    self.user_turn = None  # An audible assistant reply separates turns.
                row["text"], row["state"] = text, state
                self._touch()

    def interrupt(self, epoch, samples):
        self.advance(epoch, samples)
        for row in self.rows.values():
            if (
                row["speaker"] == "assistant"
                and row["epoch"] == epoch
                and row["state"] == "partial"
            ):
                row["state"] = "interrupted"
                row["cues"].clear()
                self._touch()

    def snapshot(self):
        return {
            "revision": self.revision,
            "ended": self.closed,
            "lookup": {"state": self.lookup_state},
            "utterances": [
                {
                    **{
                        key: row[key]
                        for key in ("id", "speaker", "text", "state", "order")
                    },
                    **(
                        {"sources": row["sources"]}
                        if row["text"] and row.get("sources")
                        else {}
                    ),
                }
                for row in self.rows.values()
                if row["text"] or row["state"] == "partial"
            ],
        }

    def close(self):
        self.clear_lookup()
        # The frontend retains its last visible snapshot until the next call.
        self.closed = True
        self.rows.clear()
        self.items.clear()
        self._touch()
