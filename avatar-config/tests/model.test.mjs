import {test} from 'node:test';
import assert from 'node:assert/strict';
import {validatePrompt,validateImage,cropRectangle,assertRevision,checkDimensions,MAX_BYTES} from '../dist/model.js';

test('empty and oversized prompts are rejected',()=>{
  assert.throws(()=>validatePrompt('  '),/Enter a prompt/);
  assert.throws(()=>validatePrompt('x'.repeat(12001)),/12,000/);
  assert.doesNotThrow(()=>validatePrompt('Be a patient conversational guide.'));
});
test('stale saves cannot replace settings from another tab',()=>{
  assert.doesNotThrow(()=>assertRevision(3,3));
  assert.throws(()=>assertRevision(4,3),/another tab/);
});
test('crop stays within portrait and landscape sources',()=>{
  for(const [width,height] of [[1600,900],[900,1600],[640,480]])
    for(const x of [0,50,100])for(const y of [0,50,100])for(const zoom of [1,2,3]) {
      const r=cropRectangle(width,height,{x,y,zoom});
      assert.ok(Math.abs(r.width/r.height-4/3)<1e-9);
      assert.ok(r.x>=0&&r.y>=0&&r.x+r.width<=width&&r.y+r.height<=height);
    }
});
test('unsupported, oversized, and malformed images fail before decoding',async()=>{
  await assert.rejects(validateImage(new File(['gif'],'a.gif',{type:'image/gif'})),/PNG or JPG/);
  await assert.rejects(validateImage(new File([new Uint8Array(MAX_BYTES+1)],'a.png',{type:'image/png'})),/10 MB/);
  await assert.rejects(validateImage(new File(['broken'],'a.png',{type:'image/png'})),/damaged/);
  await assert.rejects(validateImage(new File(['broken'],'a.jpg',{type:'image/jpeg'})),/damaged/);
});
test('PNG dimensions and animation are checked without allocating decoded pixels',async()=>{
  const bytes=new Uint8Array(45);bytes.set([137,80,78,71,13,10,26,10]);
  const view=new DataView(bytes.buffer);view.setUint32(8,13);bytes.set([73,72,68,82],12);
  view.setUint32(16,100000);view.setUint32(20,100000);
  await assert.rejects(validateImage(new File([bytes],'large.png',{type:'image/png'})),/24 million/);
  view.setUint32(16,640);view.setUint32(20,480);bytes.set([97,99,84,76],37);
  await assert.rejects(validateImage(new File([bytes],'animated.png',{type:'image/png'})),/still image/);
});
test('JPEG SOF dimensions are checked before decoding',async()=>{
  const bytes=new Uint8Array(13);bytes.set([255,216,255,192,0,9,8,255,255,255,255]);
  await assert.rejects(validateImage(new File([bytes],'large.jpg',{type:'image/jpeg'})),/24 million/);
});
test('zero dimensions and decoded pixel overflow are rejected',()=>{
  assert.throws(()=>checkDimensions(0,480),/24 million/);
  assert.throws(()=>checkDimensions(8000,8000),/24 million/);
  assert.doesNotThrow(()=>checkDimensions(640,480));
});
