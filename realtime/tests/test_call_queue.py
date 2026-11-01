import asyncio
import contextlib

from fastapi import HTTPException
import httpx
import pytest

from realtime.call_queue import CallQueue
from realtime.public_gateway import create_gateway

A, B, C = "a" * 64, "b" * 64, "c" * 64
ORIGIN = "https://demo.example.com"
PAYLOAD = {
    "avatar_id": "call01",
    "model_id": "avatarforever",
    "mode": "voice_diagnostic",
}


def test_fifo_heartbeat_expiry_and_nonrenewable_offer():
    async def run():
        now = [0.0]
        state = {"active_session": True, "avatarforever_ready": True}

        async def health():
            return dict(state)

        queue = CallQueue(health, clock=lambda: now[0])
        assert (await queue.visit(A, join=True))["position"] == 1
        assert (await queue.visit(A, join=True))["waiting"] == 1
        assert (await queue.visit(B, join=True))["position"] == 2
        assert (await queue.visit(C, join=True))["position"] == 3
        state["active_session"] = False
        now[0] = 3
        assert (await queue.visit(B))["state"] == "waiting"
        assert (await queue.visit(A))["turn_expires_in"] == 30
        now[0] = 20
        assert (await queue.visit(A))["turn_expires_in"] == 13
        now[0] = 34
        with pytest.raises(HTTPException) as exc:
            await queue.visit(A)
        assert exc.value.status_code == 410
        assert (await queue.visit(B))["state"] == "ready"
        await queue.leave(B)
        assert (await queue.visit(C))["state"] == "ready"
        await queue.leave(C)
        state["active_session"] = True
        now[0] = 40
        await queue.visit(A, join=True)
        await queue.visit(B, join=True)
        now[0] = 120
        await queue.visit(B)
        now[0] = 131
        assert (await queue.visit(B))["position"] == 1
        assert A not in queue.tickets

    asyncio.run(run())


def test_capacity_unknown_ticket_and_backend_outage_fail_closed():
    async def run():
        async def broken():
            raise httpx.ConnectError("offline")

        queue = CallQueue(broken, capacity=1)
        with pytest.raises(HTTPException) as exc:
            await queue.visit("", join=True)
        assert exc.value.status_code == 428
        assert (await queue.visit(A, join=True))["state"] == "waiting"
        with pytest.raises(HTTPException) as exc:
            await queue.visit(B, join=True)
        assert exc.value.status_code == 429
        with pytest.raises(HTTPException) as exc:
            await queue.claim(A)
        assert exc.value.status_code == 409
        with pytest.raises(HTTPException) as exc:
            await queue.visit(C)
        assert exc.value.status_code == 410

    asyncio.run(run())


@pytest.mark.parametrize("cancel_request", [False, True])
def test_public_start_cannot_bypass_fifo_or_overlap_inflight_start(cancel_request):
    async def run():
        began, finish = asyncio.Event(), asyncio.Event()
        state = {"active_session": True, "avatarforever_ready": True}
        calls = []

        async def upstream(request):
            if request.url.path == "/api/health":
                return httpx.Response(200, json=dict(state))
            assert request.url.path == "/api/sessions"
            calls.append(request)
            began.set()
            await finish.wait()
            state["active_session"] = True
            return httpx.Response(201, json={"id": "1" * 24})

        app = create_gateway(
            ORIGIN,
            "http://127.0.0.1:8080",
            "private-server-token",
            transport=httpx.MockTransport(upstream),
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url=ORIGIN,
            headers={"Origin": ORIGIN},
        ) as client:
            assert (await client.post("/api/sessions", json=PAYLOAD)).status_code == 428
            for key in (A, B):
                assert (
                    await client.post("/api/queue", headers={"X-Queue-Key": key})
                ).status_code == 200
            state["active_session"] = False
            assert (
                await client.post(
                    "/api/sessions", json=PAYLOAD, headers={"X-Queue-Key": B}
                )
            ).status_code == 409
            first = asyncio.create_task(
                client.post("/api/sessions", json=PAYLOAD, headers={"X-Queue-Key": A})
            )
            await asyncio.wait_for(began.wait(), 2)
            if cancel_request:
                first.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await first
            for key in (A, B, C):
                denied = await client.post(
                    "/api/sessions", json=PAYLOAD, headers={"X-Queue-Key": key}
                )
                assert denied.status_code in (409, 410)
            assert (
                await client.delete("/api/queue", headers={"X-Queue-Key": A})
            ).status_code == 409
            assert app.state.queue.starting == A and len(calls) == 1
            finish.set()
            if not cancel_request:
                assert (await first).status_code == 201
            for _ in range(100):
                if app.state.queue.starting is None:
                    break
                await asyncio.sleep(0.01)
            assert app.state.queue.starting is None
            assert (await client.get("/api/queue", headers={"X-Queue-Key": B})).json()[
                "state"
            ] == "waiting"
            state["active_session"] = (
                False  # Upstream teardown/reaper releases its lease.
            )
            app.state.queue.checked = float("-inf")
            next_turn = await client.get("/api/queue", headers={"X-Queue-Key": B})
            assert next_turn.json()["state"] == "ready"
            assert (
                A not in next_turn.text and "private-server-token" not in next_turn.text
            )

    asyncio.run(run())


def test_start_failure_releases_ticket_without_admitting_a_parallel_call():
    async def run():
        async def upstream(request):
            if request.url.path == "/api/health":
                return httpx.Response(
                    200, json={"active_session": False, "avatarforever_ready": True}
                )
            raise httpx.ReadTimeout("failed")

        app = create_gateway(
            ORIGIN,
            "http://127.0.0.1:8080",
            "private",
            transport=httpx.MockTransport(upstream),
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url=ORIGIN,
            headers={"Origin": ORIGIN},
        ) as client:
            for key in (A, B):
                await client.post("/api/queue", headers={"X-Queue-Key": key})
            response = await client.post(
                "/api/sessions", json=PAYLOAD, headers={"X-Queue-Key": A}
            )
            assert response.status_code == 503
            assert app.state.queue.starting is None
            assert (await client.get("/api/queue", headers={"X-Queue-Key": B})).json()[
                "state"
            ] == "ready"

    asyncio.run(run())
