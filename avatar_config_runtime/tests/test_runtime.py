import asyncio
import base64
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from fastapi import FastAPI, HTTPException
from PIL import Image
import pytest

from avatar_config_runtime.runtime import RuntimeConfiguration, install
from avatar_config_runtime.store import ConfigurationStore, InvalidConfiguration, RevisionConflict, normalize_image
from realtime.start_receipts import StartReceipts

TOKEN = 'test-only-' + 'a' * 32


def encoded(color='red'):
    out = BytesIO()
    Image.new('RGB', (80, 60), color).save(out, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(out.getvalue()).decode()


@pytest.fixture
def runtime(tmp_path):
    original = tmp_path / 'original.png'
    original.write_bytes(normalize_image(encoded()))
    catalog = SimpleNamespace(portraits={'call07': {}}, image=lambda _: original)
    manager = SimpleNamespace(catalog=catalog, lock=asyncio.Lock(), active=None,
                              create=AsyncMock(), gpu_pool=None)
    store = ConfigurationStore(tmp_path / 'versions')
    store.bootstrap('Original prompt', original)
    return RuntimeConfiguration(manager, store, avatar_id='call07', token=TOKEN)


def test_durable_versions_conflict_and_whole_pair_restore(runtime):
    async def run():
        first = runtime.store.current()[1]
        await runtime.publish(dict(prompt='New prompt', expectedRevision=1))
        reopened = ConfigurationStore(runtime.store.root)
        assert reopened.current()[1].prompt == 'New prompt'
        assert reopened.get(first.id).prompt == 'Original prompt'
        with pytest.raises(RevisionConflict):
            await runtime.publish(dict(prompt='Stale', expectedRevision=1))
        restored = await runtime.publish(dict(restoreId=first.id, expectedRevision=2))
        assert restored['revision'] == 3
        assert restored['current']['prompt'] == first.prompt
        assert restored['current']['restoredFrom'] == first.id
        assert len(restored['previous']) == 2
    asyncio.run(run())


def test_corrupt_asset_cannot_activate(runtime):
    _, first = runtime.store.current()
    runtime.store.image_path(first.image_sha256).write_bytes(b'broken')
    with pytest.raises(InvalidConfiguration):
        runtime.store.activate('New', first.image_sha256, expected_revision=1)


@pytest.mark.parametrize('payload', ['data:image/svg+xml;base64,AAAA', encoded().replace('image/png', 'image/jpeg'), 'data:image/png;base64,YmFk', None])
def test_invalid_images(payload):
    with pytest.raises(InvalidConfiguration):
        normalize_image(payload)


def test_normalization_strips_metadata_and_uses_native_geometry():
    with Image.open(BytesIO(normalize_image(encoded()))) as image:
        assert image.size == (640, 480)
        assert image.mode == 'RGB'
        assert not image.info


def test_token_origin_and_payload_boundaries(runtime):
    async def run():
        app = FastAPI()
        runtime.install_routes(app)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://localhost') as client:
            assert (await client.get('/api/avatar-config')).status_code == 401
            auth = {'Authorization': 'Bearer ' + TOKEN}
            assert (await client.get('/api/avatar-config', headers=auth)).status_code == 200
            assert (await client.post('/api/avatar-config', headers={**auth, 'Origin': 'https://evil.test'}, json={})).status_code == 403
            assert (await client.post('/api/avatar-config', headers=auth, json={'path': '/etc/passwd'})).status_code == 422
            assert (await client.post('/api/avatar-config', headers=auth, content='oops')).status_code == 400
            assert (await client.post('/api/avatar-config', headers=auth, json={'prompt':'New', 'expectedRevision':True})).status_code == 422
            assert runtime.store.current()[0] == 1
    asyncio.run(run())


def test_unverified_and_busy_image_changes_keep_current(runtime):
    async def run():
        request = dict(prompt='New', image=encoded('blue'), expectedRevision=1)
        with pytest.raises(HTTPException, match='renderer switching'):
            await runtime.publish(request)
        runtime.image_switch_verified = True
        runtime.manager.active = 'existing-call'
        runtime.prepare_image = AsyncMock()
        with pytest.raises(HTTPException, match='active call'):
            await runtime.publish(request)
        runtime.prepare_image.assert_not_called()
        assert runtime.store.current()[1].prompt == 'Original prompt'
        await runtime.publish(dict(prompt='Prompt only', expectedRevision=1))
        assert runtime.manager.active == 'existing-call'
    asyncio.run(run())


def test_preparation_failure_and_rollback_restore_pair(runtime):
    async def run():
        original = runtime.store.current()[1]
        runtime.image_switch_verified = True
        runtime.prepare_image = AsyncMock(side_effect=RuntimeError('renderer failed'))
        request = dict(prompt='Blue prompt', image=encoded('blue'), expectedRevision=1)
        with pytest.raises(RuntimeError):
            await runtime.publish(request)
        assert runtime.store.current() == (1, original)
        runtime.prepare_image.side_effect = None
        await runtime.publish(request)
        blue = runtime.store.current()[1]
        assert blue.image_sha256 != original.image_sha256
        await runtime.publish(dict(restoreId=original.id, expectedRevision=2))
        _, restored = runtime.store.current()
        assert (restored.prompt, restored.image_sha256) == (original.prompt, original.image_sha256)
        runtime.prepare_image.assert_awaited_with(original.image_sha256, blue.image_sha256)
    asyncio.run(run())


def test_call_pin_recovery_and_catalog_isolation(runtime):
    async def run():
        # Reinstall admission around a simulated constructor using the real binding
        # hooks. StartReceipts is the same recovery mechanism used by both call ports.
        captured = []
        async def create(*args):
            voice = SimpleNamespace(instructions='Default')
            runtime.configure_voice(voice)
            catalog = runtime.worker_catalog(runtime.manager.catalog)
            captured.append((voice, catalog))
            return len(captured)
        runtime.manager.create = create
        runtime.install_admission()
        voices = {'profile': {'provider': 'gpt-live-1.1'}}
        receipts = StartReceipts()
        context = runtime.trial_request.set(True)
        try:
            first = runtime.store.current()[1]
            one = await receipts.run('a'*64, {}, lambda: runtime.manager.create('call07', voices, 'avatarforever'))
            await runtime.publish(dict(prompt='Second prompt', expectedRevision=1))
            assert await receipts.run('a'*64, {}) == one
            assert len(captured) == 1
            assert captured[0][0].instructions == 'Original prompt'
            assert captured[0][0].configuration_version == first.id
            assert captured[0][1].image('call07') != runtime.manager.catalog.image('call07')
            await receipts.run('b'*64, {}, lambda: runtime.manager.create('call07', voices, 'avatarforever'))
            assert captured[1][0].instructions == 'Second prompt'
        finally:
            runtime.trial_request.reset(context)
        await runtime.manager.create('call07', voices, 'avatarforever')
        assert captured[-1][0].instructions == 'Default'
        assert captured[-1][1] is runtime.manager.catalog
        assert runtime.pinned.get() is None
        await receipts.close()
    asyncio.run(run())


def test_simultaneous_publication_has_one_winner(runtime):
    async def run():
        results = await asyncio.gather(*(
            runtime.publish(dict(prompt=prompt, expectedRevision=1)) for prompt in ['One', 'Two']
        ), return_exceptions=True)
        assert sum(isinstance(r, RevisionConflict) for r in results) == 1
        assert runtime.store.current()[0] == 2
    asyncio.run(run())


def test_missing_opt_in_does_not_change_runtime(runtime, tmp_path):
    original = runtime.manager.create
    assert install(FastAPI(), runtime.manager, 'Default', tmp_path / 'absent.json') is None
    assert runtime.manager.create is original


def test_preparation_reuses_worker_and_returns_to_current(runtime):
    import numpy as np
    async def run():
        original = runtime.store.current()[1]
        candidate = runtime.store.put_image(normalize_image(encoded('blue')))
        calls = []
        gpu = SimpleNamespace(closed=False, generate=AsyncMock(return_value=SimpleNamespace(frames=np.zeros((1,480,640,3),dtype=np.uint8))))
        async def acquire(owner, path):
            calls.append(path)
            return gpu
        pool = SimpleNamespace(worker=gpu, owner=None, snapshot=lambda: {'loaded':True},
                               acquire=AsyncMock(side_effect=acquire), release=AsyncMock())
        runtime.manager.gpu_pool = pool
        await runtime.prepare_image(candidate, original.image_sha256)
        assert pool.worker is gpu
        assert calls == [runtime.store.image_path(candidate), runtime.store.image_path(original.image_sha256)]
        assert pool.release.await_count == 2
        # A failed candidate still restores the previous reference when IPC is healthy.
        calls.clear()
        gpu.generate.side_effect = [RuntimeError('failed'), SimpleNamespace(frames=np.zeros((1,480,640,3),dtype=np.uint8))]
        with pytest.raises(RuntimeError):
            await runtime.prepare_image(candidate, original.image_sha256)
        assert calls[-1] == runtime.store.image_path(original.image_sha256)
        assert runtime.store.current()[1] == original
        pool.snapshot = lambda: {'loaded':False}
        pool.acquire.reset_mock()
        with pytest.raises(HTTPException):
            await runtime.prepare_image(candidate, original.image_sha256)
        pool.acquire.assert_not_called()
    asyncio.run(run())


def test_disconnected_publisher_finishes_transaction(runtime):
    async def run():
        app = FastAPI()
        runtime.install_routes(app)
        started, finish = asyncio.Event(), asyncio.Event()
        runtime.image_switch_verified = True
        async def prepare(*args):
            started.set()
            await finish.wait()
        runtime.prepare_image = prepare
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://localhost') as client:
            post = asyncio.create_task(client.post('/api/avatar-config',headers={'Authorization':'Bearer '+TOKEN},json=dict(prompt='Committed after disconnect',image=encoded('blue'),expectedRevision=1)))
            await started.wait()
            post.cancel()
            with pytest.raises(asyncio.CancelledError):
                await post
            finish.set()
            await runtime.close()
            assert runtime.store.current()[0] == 2
            assert runtime.store.current()[1].prompt == 'Committed after disconnect'
            with pytest.raises(RevisionConflict):
                await runtime.publish(dict(prompt='Retry',expectedRevision=1))
    asyncio.run(run())
