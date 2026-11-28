import test from 'node:test';
import assert from 'node:assert/strict';
import {request} from 'node:http';
import {createDashboardServer} from '../server.mjs';
const token='test-token-'+'a'.repeat(32);
async function withServer(options,run){
  const server=createDashboardServer(options);
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const origin=`http://127.0.0.1:${server.address().port}`;
  try{await run(origin);}finally{await new Promise(resolve=>server.close(resolve));}
}
test('connection secret stays server-side and only fixed runtime route is used',async()=>{
  const calls=[];
  await withServer({readConnection:async()=>({token}),fetchRuntime:async(url,options)=>{calls.push({url,options});return Response.json({revision:1});}},async origin=>{
    const result=await fetch(origin+'/api/configuration');
    assert.deepEqual(await result.json(),{revision:1});
    assert.equal(calls[0].url,'http://127.0.0.1:18837/api/avatar-config');
    assert.equal(calls[0].options.headers.Authorization,'Bearer '+token);
    assert.equal((await fetch(origin+'/local/runtime.json')).status,404);
    const write=await fetch(origin+'/api/configuration',{method:'POST',headers:{Origin:origin,'Content-Type':'application/json'},body:'{"prompt":"New"}'});
    assert.equal(write.status,200);
    assert.equal(calls[1].options.body.toString(),'{"prompt":"New"}');
  });
});
test('rejects cross-origin and rebinding requests before accessing connection',async()=>{
  await withServer({readConnection:async()=>{assert.fail('credentials must not be read');}},async origin=>{
    for(const headers of [{Host:'evil.test'},{Origin:'https://evil.test'},{'Sec-Fetch-Site':'cross-site'}]){
      const status=await new Promise((resolve,reject)=>{const req=request(origin+'/api/configuration',{headers},res=>{res.resume();resolve(res.statusCode);});req.on('error',reject);req.end();});
      assert.equal(status,403);
    }
    assert.equal((await fetch(origin+'/api/configuration',{method:'POST',body:'{}'})).status,403);
  });
});
test('unconfigured checkouts are read-only and runtime errors contain no secrets',async()=>{
  await withServer({readConnection:async()=>{throw Object.assign(new Error(token),{code:'ENOENT'});}},async origin=>{
    const response=await fetch(origin+'/api/configuration');assert.equal(response.status,404);assert.ok(!(await response.text()).includes(token));
  });
  await withServer({readConnection:async()=>({token}),fetchRuntime:async()=>{throw Error(token);}},async origin=>{
    const response=await fetch(origin+'/api/configuration');assert.equal(response.status,503);assert.ok(!(await response.text()).includes(token));
  });
});
