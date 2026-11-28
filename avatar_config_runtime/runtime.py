"""Opt-in configuration for one private GPT-Live deployment, never the catalog."""
from __future__ import annotations

import asyncio
import base64
from contextvars import ContextVar
import hmac
import json
from pathlib import Path
import secrets

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .store import ConfigurationStore, InvalidConfiguration, RevisionConflict, normalize_image, validate_prompt

MAX_REQUEST = 14 * 1024 * 1024


class CallCatalog:
    """A single worker's image override; the shared/public catalog stays untouched."""
    def __init__(self, catalog, avatar_id, image):
        self.catalog, self.avatar_id, self.path = catalog, avatar_id, image

    def image(self, avatar_id):
        return self.path if avatar_id == self.avatar_id else self.catalog.image(avatar_id)

    def __getattr__(self, name):
        return getattr(self.catalog, name)


class RuntimeConfiguration:
    def __init__(self, manager, store, *, avatar_id, token, image_switch_verified=False):
        if not isinstance(token, str) or len(token) < 32:
            raise ValueError('A separate configuration token of at least 32 characters is required')
        if avatar_id not in manager.catalog.portraits:
            raise ValueError('Configuration must target an existing private avatar')
        self.manager, self.store = manager, store
        self.avatar_id, self.token = avatar_id, token
        self.image_switch_verified = image_switch_verified
        self.trial_request = ContextVar('avatar_config_trial', default=False)
        self.pinned = ContextVar('avatar_config_version', default=None)
        self.write_lock = asyncio.Lock()
        self.jobs = set()
        self.call_source = ContextVar('avatar_call_source', default='web')
        self.logs = None
        self.install_admission()

    def install_admission(self):
        original = self.manager.create

        async def create(avatar_id, voice_selection=None, model_id='mock'):
            selected = (self.trial_request.get() and avatar_id == self.avatar_id
                        and model_id == 'avatarforever' and voice_selection
                        and voice_selection['profile'].get('provider') == 'gpt-live-1.1')
            version = self.store.current()[1] if selected else None
            context = self.pinned.set(version)
            try:
                return await original(avatar_id, voice_selection, model_id)
            finally:
                self.pinned.reset(context)

        self.manager.create = create

    def worker_catalog(self, catalog):
        version = self.pinned.get()
        return CallCatalog(catalog, self.avatar_id, self.store.image_path(version.image_sha256)) if version else catalog

    def configure_voice(self, voice):
        if version := self.pinned.get():
            voice.instructions = version.prompt
            # Private in-process audit only; never added to public snapshots/room data.
            voice.configuration_version = version.id

    def configure_worker(self, worker):
        if self.logs is None:
            return
        from .recording import CallRecorder
        recorder = CallRecorder(worker, self.logs, self.call_source.get())
        worker.call_recorder = recorder
        worker.voice.recording_enabled = True
        start, close = worker.start, worker._close

        async def recorded_start():
            if getattr(worker, 'parallel_startup', False):
                return await start()
            try:
                await recorder.start()
            except Exception:
                recorder.error = 'Recording connection could not start'
            await start()

        async def recorded_close():
            try:
                await close()
            finally:
                await recorder.close()

        worker.start, worker._close = recorded_start, recorded_close

    def authorize(self, request):
        supplied = request.headers.get('authorization', '')
        if not hmac.compare_digest(supplied.encode(), ('Bearer ' + self.token).encode()):
            raise HTTPException(401, 'Configuration authorization required')
        # Browser clients go through the same-origin dashboard server. This API
        # accepts only server-to-server calls, not cross-origin browser writes.
        if request.headers.get('origin'):
            raise HTTPException(403, 'Use the dashboard configuration connection')

    def record(self, version):
        raw = self.store.image_path(version.image_sha256).read_bytes()
        return dict(id=version.id, date=version.created_at[:10], detail=version.source,
                    prompt=version.prompt, greeting='', image=dict(
                        src='data:image/png;base64,' + base64.b64encode(raw).decode(),
                        label='Driving image', note=''), restoredFrom=version.restored_from)

    def snapshot(self):
        revision, current = self.store.current()
        return dict(revision=revision, current=self.record(current),
                    previous=[self.record(v) for v in self.store.history()],
                    imageUploadsEnabled=self.image_switch_verified,
                    activeCall=self.manager.active is not None)

    async def prepare_image(self, candidate, current):
        """Exercise candidate and return to current with the existing loaded weights."""
        import numpy as np
        from realtime.contracts import AudioChunk, FRAME_SAMPLES
        pool = self.manager.gpu_pool
        loaded = pool.worker
        if not pool.snapshot()['loaded'] or pool.owner is not None:
            raise HTTPException(409, 'The existing renderer must be ready and idle')
        # Never cold-load a replacement model during a configuration write.
        async def exercise(sha):
            if pool.worker is not loaded or loaded.closed:
                raise RuntimeError('The existing renderer is no longer available')
            owner = 'configuration-' + secrets.token_hex(12)
            try:
                gpu = await pool.acquire(owner, self.store.image_path(sha))
                audio = AudioChunk('configuration-check', 1, 0,
                                   np.zeros(FRAME_SAMPLES, dtype=np.int16), FRAME_SAMPLES)
                result = await gpu.generate(audio)
                if result is None or result.frames.shape != (1, 480, 640, 3):
                    raise RuntimeError('Renderer did not return a valid frame')
            finally:
                await pool.release(owner)
        try:
            await exercise(candidate)
        finally:
            # Failure leaves activation unchanged. Do not reuse a failed process;
            # normal call admission/operator recovery owns any later cold load.
            if pool.worker is loaded and not loaded.closed:
                await exercise(current)

    async def publish(self, body):
        if not isinstance(body, dict) or set(body) - {'prompt', 'image', 'expectedRevision', 'restoreId'}:
            raise InvalidConfiguration('Invalid deployment request')
        expected = body.get('expectedRevision')
        if type(expected) is not int or expected < 1:
            raise InvalidConfiguration('A valid deployment revision is required')
        async with self.write_lock:
            revision, current = self.store.current()
            if revision != expected:
                raise RevisionConflict('Deployment changed. Reload before publishing.')
            if 'restoreId' in body:
                if set(body) != {'restoreId', 'expectedRevision'} or not isinstance(body['restoreId'], str):
                    raise InvalidConfiguration('Invalid restore request')
                target = self.store.get(body['restoreId'])
                prompt, sha = target.prompt, target.image_sha256
            else:
                prompt = validate_prompt(body.get('prompt'))
                if body.get('image') is None:
                    sha = current.image_sha256
                else:
                    if not self.image_switch_verified:
                        raise HTTPException(409, 'Image changes await the renderer switching check')
                    raw = await asyncio.to_thread(normalize_image, body['image'])
                    sha = self.store.put_image(raw)
            async with self.manager.lock:
                if sha != current.image_sha256:
                    if not self.image_switch_verified:
                        raise HTTPException(409, 'Image changes await the renderer switching check')
                    if self.manager.active is not None:
                        raise HTTPException(409, 'End the active call before preparing a new image')
                    await self.prepare_image(sha, current.image_sha256)
                self.store.activate(prompt, sha, expected_revision=expected,
                                    source='restore' if 'restoreId' in body else 'publish',
                                    restored_from=body.get('restoreId'))
            return self.snapshot()

    def install_routes(self, app):
        @app.get('/api/avatar-config')
        async def read(request: Request):
            query = request.query_params
            operation = query.get('op')
            if operation == 'download':
                if self.logs is None or not self.logs.verify(query.get('id'), query.get('file'), query.get('expires'), query.get('signature', '')):
                    raise HTTPException(403, 'Download link expired or invalid')
                try:
                    path, media = self.logs.file(query['id'], query['file'])
                except (KeyError, ValueError):
                    raise HTTPException(404, 'Recording not found') from None
                return FileResponse(path, media_type=media, filename=query['id']+'-'+path.name,
                                    headers={'Cache-Control':'private, no-store', 'Referrer-Policy':'no-referrer', 'X-Content-Type-Options':'nosniff'})
            self.authorize(request)
            if operation in {'calls', 'call', 'ticket'}:
                if self.logs is None:
                    raise HTTPException(503, 'Call recording is not enabled')
                try:
                    if operation == 'calls':
                        offset = int(query.get('offset', '0'))
                        if not 0 <= offset <= 1000000:
                            raise ValueError()
                        return await asyncio.to_thread(self.logs.listing, offset)
                    if operation == 'call':
                        return await asyncio.to_thread(self.logs.get, query.get('id'))
                    return await asyncio.to_thread(self.logs.ticket, query.get('id'), query.get('file'))
                except ValueError:
                    raise HTTPException(400, 'Invalid call request') from None
                except KeyError:
                    raise HTTPException(404, 'Recording not found') from None
            if operation:
                raise HTTPException(400, 'Unknown operation')
            return self.snapshot()

        @app.post('/api/avatar-config')
        async def write(request: Request):
            self.authorize(request)
            payload = bytearray()
            async for chunk in request.stream():
                payload.extend(chunk)
                if len(payload) > MAX_REQUEST:
                    raise HTTPException(413, 'Image is too large')
            try:
                body = json.loads(payload)
            except (ValueError, UnicodeDecodeError):
                raise HTTPException(400, 'Invalid JSON') from None
            # Finish a publication once admitted, including when the browser loses
            # its response. CAS prevents a retried POST from publishing twice.
            job = asyncio.create_task(self.publish(body))
            self.jobs.add(job)
            def completed(task):
                self.jobs.discard(task)
                if not task.cancelled():
                    task.exception()
            job.add_done_callback(completed)
            try:
                return await asyncio.shield(job)
            except InvalidConfiguration as exc:
                raise HTTPException(422, str(exc)) from None
            except RevisionConflict as exc:
                raise HTTPException(409, str(exc)) from None
            except HTTPException:
                raise
            except Exception:
                raise HTTPException(503, 'Image preparation failed. Current deployment was kept.') from None

    async def close(self):
        await asyncio.gather(*self.jobs, return_exceptions=True)


def install(app, manager, default_prompt, config_path: Path):
    """Missing file leaves existing serving behavior completely unchanged."""
    if not config_path.exists():
        return None
    config = json.loads(config_path.read_text())
    # Fixed local deployment scope, configured by the host operator, not a client ID.
    store = ConfigurationStore(config_path.parent / 'versions')
    avatar_id = config['avatar_id']
    store.bootstrap(default_prompt, manager.catalog.image(avatar_id))
    runtime = RuntimeConfiguration(manager, store, avatar_id=avatar_id,
                                   token=config['admin_token'],
                                   image_switch_verified=config.get('image_switch_verified') is True)
    if config.get('record_calls') is True:
        from .call_logs import CallLogs
        runtime.logs = CallLogs(config_path.parent / 'calls', config['admin_token'])
    runtime.api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    runtime.api.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', '[::1]'])
    runtime.install_routes(runtime.api)
    return runtime
