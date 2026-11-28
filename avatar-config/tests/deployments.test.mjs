import test from 'node:test';
import assert from 'node:assert/strict';
import {callLinks,validateDeployments} from '../dist/deployments.js';
const record={id:'example-current',date:'2026-11-26',detail:'640 × 480',prompt:'Let the caller speak first.',greeting:'The FaceTime call is now connected.',image:{src:null,label:'Example portrait',note:'Not supplied'}};
const snapshot={previewUrl:'http://127.0.0.1:18837/',facetimeNumber:'+12025550123',recordedAt:'2026-11-27',current:record,previous:[{...record,id:'example-previous',detail:'512 × 384'}]};
test('recorded deployments retain actual prompt and greeting separately',()=>{
  const data=validateDeployments(snapshot);
  assert.equal(data.current.detail,'640 × 480');
  assert.equal(data.previous[0].detail,'512 × 384');
  assert.equal(data.current.prompt,data.previous[0].prompt);
  assert.match(data.current.prompt,/Let the caller speak first/);
  assert.match(data.current.greeting,/FaceTime call is now connected/);
  assert.equal(data.current.image.src,null);
});
test('links open the existing web call and native FaceTime video handler',()=>{
  assert.deepEqual(callLinks(snapshot),{preview:'http://127.0.0.1:18837/',facetime:'facetime:+12025550123'});
});
test('rejects executable, remote or credential-bearing preview links and invalid phone numbers',()=>{
  for(const previewUrl of ['javascript:alert(1)','https://example.com','http://user:pass@localhost/']) assert.throws(()=>callLinks({...snapshot,previewUrl}));
  for(const facetimeNumber of ['123','+12025550123?x=1','someone@example.com']) assert.throws(()=>callLinks({...snapshot,facetimeNumber}));
});
test('rejects duplicate records, missing prompts, and remote or executable images',()=>{
  assert.throws(()=>validateDeployments({...snapshot,previous:[snapshot.current]}));
  assert.throws(()=>validateDeployments({...snapshot,current:{...snapshot.current,prompt:''}}));
  for(const src of ['https://example.com/private.png','data:image/svg+xml;base64,PHN2Zz4=']) assert.throws(()=>validateDeployments({...snapshot,current:{...snapshot.current,image:{...snapshot.current.image,src}}}));
});

test('fresh checkouts have no private deployment data or callable number',()=>{
  const data={previewUrl:'http://127.0.0.1:18837/',facetimeNumber:null,recordedAt:null,current:null,previous:[]};
  assert.equal(validateDeployments(data),data);
  assert.equal(callLinks(data).facetime,null);
});

test('live connection retains recorded history without making snapshots restorable',async()=>{
  const {deploymentHistory}=await import('../dist/deployments.js');
  const live={current:{...record,id:'live-current'},previous:[{...record,id:'live-previous'}]};
  assert.deepEqual(deploymentHistory(live,snapshot).map(({record,restorable})=>[record.id,restorable]),[['live-previous',true],['example-previous',false]]);
  assert.deepEqual(deploymentHistory(null,snapshot).map(item=>item.restorable),[false]);
});
