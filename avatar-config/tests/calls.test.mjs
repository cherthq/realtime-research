import test from 'node:test';
import assert from 'node:assert/strict';
import {handleCalls} from '../api/calls.js';
const origin='https://dashboard.example',id='a'.repeat(24);
const env={AVATAR_RUNTIME_URL:'https://runtime.example/api/avatar-config',AVATAR_RUNTIME_TOKEN:'test-'.repeat(10)};
test('call logs use fixed upstream, scoped credentials and bounded queries',async()=>{
  let seen;const options={env,fetch:async(url,settings)=>{seen={url,settings};return Response.json({calls:[]});}};
  const response=await handleCalls(new Request(origin+'/api/calls?offset=50'),options);
  assert.equal(response.status,200);assert.equal(seen.url.searchParams.get('offset'),'50');assert.equal(seen.url.origin,'https://runtime.example');assert.equal(seen.settings.headers.Authorization,'Bearer '+env.AVATAR_RUNTIME_TOKEN);
  for(const query of ['?id=../secret','?id='+id+'&download=secret','?offset=-1'])assert.equal((await handleCalls(new Request(origin+'/api/calls'+query),options)).status,400);
  assert.equal((await handleCalls(new Request(origin+'/api/calls',{headers:{Origin:'https://evil.example'}}),options)).status,403);
});
test('large media downloads redirect only to a scoped, expiring runtime ticket',async()=>{
 const response=await handleCalls(new Request(origin+'/api/calls?id='+id+'&download=video'),{env,fetch:async()=>Response.json({id,file:'video',expires:'2000000000',signature:'b'.repeat(64)})});
 assert.equal(response.status,302);const url=new URL(response.headers.get('location'));assert.equal(url.origin,'https://runtime.example');assert.equal(url.searchParams.get('op'),'download');assert.ok(!url.href.includes(env.AVATAR_RUNTIME_TOKEN));assert.equal(response.headers.get('cache-control'),'private, no-store');
 const bad=await handleCalls(new Request(origin+'/api/calls?id='+id+'&download=video'),{env,fetch:async()=>Response.json({id,file:'audio',expires:'1',signature:'b'.repeat(64)})});assert.equal(bad.status,503);
});
