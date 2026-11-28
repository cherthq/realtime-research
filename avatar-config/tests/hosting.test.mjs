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

test('hosting build keeps private records out of static output',async()=>{
  const {mkdtemp,cp,readdir,readFile,rm}=await import('node:fs/promises');
  const {tmpdir}=await import('node:os');const {join}=await import('node:path');
  const {spawnSync}=await import('node:child_process');const {createHash}=await import('node:crypto');
  const temporary=await mkdtemp(join(tmpdir(),'avatar-hosting-test-'));
  try {
    for(const path of ['build.mjs','dist','lib'])await cp(new URL('../'+path,import.meta.url),join(temporary,path),{recursive:true});
    const src='data:image/png;base64,aW1hZ2UtZml4dHVyZQ==';
    const imageHash=createHash('sha256').update(Buffer.from(src.split(',')[1],'base64')).digest('hex');
    const record={id:'fixture',date:'2026-11-28',detail:'fixture',prompt:'Private fixture prompt',greeting:'',image:{src:null,label:'Fixture',note:'',sha256:imageHash}};
    const metadata={previewUrl:'http://127.0.0.1:18837/',facetimeNumber:null,recordedAt:'2026-11-28',current:record,previous:[]};
    const live={current:{...record,image:{...record.image,src}},previous:[]};
    const source=`globalThis.fetch=async()=>Response.json(${JSON.stringify(live)}); await import('./build.mjs');`;
    const built=spawnSync(process.execPath,['--input-type=module','-e',source],{cwd:temporary,env:{...process.env,...env,AVATAR_DEPLOYMENT_RECORDS:JSON.stringify(metadata)},encoding:'utf8'});
    assert.equal(built.status,0,built.stderr);
    assert.deepEqual((await readdir(join(temporary,'public'))).sort(),['app.js','deployments.js','index.html','model.js','style.css']);
    for(const name of await readdir(join(temporary,'public'))){const body=await readFile(join(temporary,'public',name),'utf8');assert.ok(!body.includes(record.prompt));assert.ok(!body.includes(src));assert.ok(!body.includes(env.AVATAR_RUNTIME_TOKEN));}
    const privateRecords=JSON.parse(await readFile(join(temporary,'private','recorded.json'),'utf8'));
    assert.equal(privateRecords.current.image.src,src);
  }finally{await rm(temporary,{recursive:true,force:true});}
});
