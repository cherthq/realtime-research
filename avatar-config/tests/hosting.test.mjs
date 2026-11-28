import test from 'node:test';
import assert from 'node:assert/strict';
import {checkRequest,readBody,runtimeRequest,runtimeSettings,MAX_BODY} from '../lib/runtime-client.mjs';
const origin='https://avatar.example';
const env={AVATAR_RUNTIME_URL:'https://runtime.example/api/avatar-config',AVATAR_RUNTIME_TOKEN:'test-only-'+'a'.repeat(32)};
test('hosted writes require same-origin JSON while server reads remain available',()=>{
  assert.equal(checkRequest(new Request(origin+'/api/configuration')),null);
  for(const options of [
    {method:'POST',body:'{}'},
    {method:'POST',headers:{Origin:'https://evil.example','Content-Type':'application/json'},body:'{}'},
    {headers:{'Sec-Fetch-Site':'cross-site'}},
    {method:'DELETE'}
  ])assert.ok(checkRequest(new Request(origin+'/api/configuration',options)));
  assert.equal(checkRequest(new Request(origin+'/api/configuration',{method:'POST',headers:{Origin:origin,'Content-Type':'application/json'},body:'{}'})),null);
});
test('runtime destination is fixed by server settings and only scoped authorization is forwarded',async()=>{
  let called;
  await runtimeRequest('GET',undefined,{env,fetch:async(...args)=>{called=args;return Response.json({revision:1});}});
  assert.equal(called[0],env.AVATAR_RUNTIME_URL);
  assert.equal(called[1].headers.Authorization,'Bearer '+env.AVATAR_RUNTIME_TOKEN);
  assert.equal(called[1].redirect,'error');
  for(const value of ['http://runtime.example/api/avatar-config','https://user:password@runtime.example/api/avatar-config','https://runtime.example/api/terminals','https://runtime.example/api/avatar-config?token=x'])assert.throws(()=>runtimeSettings({...env,AVATAR_RUNTIME_URL:value}));
});
test('hosted body limits reject excess data before the runtime request',async()=>{
  assert.equal(await readBody(new Request(origin,{method:'POST',body:'{"prompt":"Hi"}'})),'{"prompt":"Hi"}');
  await assert.rejects(readBody(new Request(origin,{method:'POST',headers:{'Content-Length':String(MAX_BODY+1)},body:'x'})));
  await assert.rejects(readBody(new Request(origin,{method:'POST',body:'x'.repeat(MAX_BODY+1)})));
});
