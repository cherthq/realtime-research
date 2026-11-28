import {emptyCrop,cropRectangle,checkDimensions,validateImage,openLocalStore,loadConfiguration,saveConfiguration} from './model.js';
import {loadDeployments} from './deployments.js';

const $=id=>document.getElementById(id);
const form=$('editor'),prompt=$('prompt'),fileInput=$('image-file'),picker=$('image-picker'),preview=$('image-preview');
const save=$('save'),saveStatus=$('save-status'),imageStatus=$('image-status'),error=$('error'),reload=$('reload');
const ranges=['zoom','x','y'].map($);
let db, revision=0, image=null, bitmap=null, sourceUrl='',previewUrl='';
let runtime=null,publishing=false;
let dirty=false,saving=false,loading=false,ready=false,conflict=false,generation=0;

function showError(message) { error.textContent=message;error.hidden=!message; }
function updateControls() {
  const blocked=!ready||saving||publishing;
  form.setAttribute('aria-busy',String(blocked||loading));
  prompt.disabled=blocked;fileInput.disabled=blocked;picker.disabled=blocked;
  $('remove-image').disabled=blocked;
  ranges.forEach(input=>input.disabled=blocked||loading);
  save.disabled=blocked||loading||!dirty||conflict;
  save.textContent=saving?'Saving…':'Save locally';
  $('publish').hidden=!runtime;
  $('publish').disabled=blocked||loading||!prompt.value.trim()||!image;
  $('publish').textContent=publishing?'Preparing…':'Publish';
  $('refresh-deployments').disabled=publishing;
  document.querySelectorAll('.restore').forEach(button=>button.disabled=publishing);
}
function changed() {dirty=true;saveStatus.textContent='Unsaved changes';updateControls();}
function releaseImage() {
  bitmap=null;
  if(sourceUrl)URL.revokeObjectURL(sourceUrl);
  if(previewUrl)URL.revokeObjectURL(previewUrl);
  sourceUrl='';previewUrl='';
}
function displayImage(blob) {
  if(previewUrl)URL.revokeObjectURL(previewUrl);
  previewUrl=blob?URL.createObjectURL(blob):'';
  preview.hidden=!blob;
  if(blob)preview.src=previewUrl;else preview.removeAttribute('src');
  $('empty-image').hidden=Boolean(blob);$('image-hover').hidden=!blob;
  $('crop-panel').hidden=!blob;$('remove-image').hidden=!blob;
  picker.classList.toggle('has-image',Boolean(blob));
}
async function decode(blob) {
  const url=URL.createObjectURL(blob),img=new Image();img.src=url;
  try {await img.decode();checkDimensions(img.naturalWidth,img.naturalHeight);return {img,url};}
  catch(e) {URL.revokeObjectURL(url);throw new Error(e.message?.includes('24 million')?e.message:'Could not read this image. Choose another PNG or JPG.');}
}
function makeCrop(img,crop) {
  const rect=cropRectangle(img.naturalWidth,img.naturalHeight,crop);
  const canvas=document.createElement('canvas');canvas.width=640;canvas.height=480;
  const ctx=canvas.getContext('2d');
  if(!ctx)throw new Error('Image preview is unavailable in this browser.');
  ctx.fillStyle='#fff';ctx.fillRect(0,0,640,480);
  ctx.drawImage(img,rect.x,rect.y,rect.width,rect.height,0,0,640,480);
  return new Promise((resolve,reject)=>canvas.toBlob(blob=>blob?resolve(blob):reject(new Error('Could not prepare the image. Try again.')),'image/png'));
}
async function choose(file) {
  if(!ready||saving||publishing||!file)return;
  const request=++generation;loading=true;showError('');imageStatus.textContent='Preparing image…';updateControls();
  let decoded;
  try {
    await validateImage(file);decoded=await decode(file);
    const crop=emptyCrop(),result=await makeCrop(decoded.img,crop);
    if(request!==generation){URL.revokeObjectURL(decoded.url);return;}
    releaseImage();bitmap=decoded.img;sourceUrl=decoded.url;
    image={source:file,preview:result,name:file.name,crop};
    ranges.forEach(input=>input.value=String(crop[input.id]));
    displayImage(result);imageStatus.textContent='640 × 480 · Still preview';changed();
  } catch(e) {
    if(decoded?.url)URL.revokeObjectURL(decoded.url);
    if(request===generation){showError(e.message);imageStatus.textContent=image?'Current image kept':'';}
  } finally {if(request===generation){loading=false;updateControls();}}
}
async function adjustCrop() {
  if(!image||!bitmap||saving||publishing)return;
  const request=++generation;
  const crop={x:Number($('x').value),y:Number($('y').value),zoom:Number($('zoom').value)};
  loading=true;save.disabled=true;imageStatus.textContent='Preparing crop…';
  try {
    const result=await makeCrop(bitmap,crop);
    if(request!==generation)return;
    image={...image,crop,preview:result};displayImage(result);
    imageStatus.textContent='640 × 480 · Still preview';showError('');changed();
  } catch(e) {if(request===generation){showError(e.message);ranges.forEach(input=>input.value=String(image.crop[input.id]));}}
  finally {if(request===generation){loading=false;updateControls();}}
}
async function loadSaved() {
  ready=false;updateControls();showError('');
  try {
    if(!db)db=await openLocalStore();
    const configuration=await loadConfiguration(db);
    const decoded=configuration.image?await decode(configuration.image.source):null;
    releaseImage();image=configuration.image;revision=configuration.revision;
    prompt.value=configuration.prompt;
    if(decoded){bitmap=decoded.img;sourceUrl=decoded.url;}
    ranges.forEach(input=>input.value=String((image?.crop??emptyCrop())[input.id]));
    displayImage(image?.preview??null);
    imageStatus.textContent=image?'640 × 480 · Still preview':'';
    dirty=false;conflict=false;reload.hidden=true;ready=true;
    saveStatus.textContent=revision?'Saved locally':'';
  } catch(e) {showError(e.message);reload.hidden=false;reload.textContent='Retry loading';saveStatus.textContent='';}
  updateControls();
}
prompt.addEventListener('input',changed);
picker.addEventListener('click',()=>fileInput.click());
fileInput.addEventListener('change',()=>{void choose(fileInput.files[0]);fileInput.value='';});
for(const event of ['dragenter','dragover'])picker.addEventListener(event,e=>{e.preventDefault();if(ready&&!saving&&!publishing)picker.classList.add('dragging');});
picker.addEventListener('dragleave',()=>picker.classList.remove('dragging'));
picker.addEventListener('drop',e=>{e.preventDefault();picker.classList.remove('dragging');void choose(e.dataTransfer.files[0]);});
ranges.forEach(input=>input.addEventListener('input',()=>void adjustCrop()));
$('remove-image').addEventListener('click',()=>{
  if(!ready||saving||publishing)return;
  generation++;loading=false;releaseImage();image=null;displayImage(null);imageStatus.textContent='';showError('');changed();
});
reload.addEventListener('click',()=>{generation++;loading=false;void (async()=>{await loadSaved();await refreshDeployments();if(runtime&&!revision&&!dirty&&!image&&!prompt.value)await editDeployment(runtime.current);})();});
form.addEventListener('submit',async e=>{
  e.preventDefault();if(!ready||saving||publishing||loading||!dirty||conflict)return;
  saving=true;showError('');updateControls();
  try {
    const saved=await saveConfiguration(db,{prompt:prompt.value,image},revision);
    revision=saved.revision;dirty=false;saveStatus.textContent='Saved locally';
  } catch(e) {
    showError(e.message);
    if(e.message.includes('another tab')){conflict=true;reload.textContent='Discard edits and load saved';reload.hidden=false;}
  } finally {saving=false;updateControls();}
});
async function refreshDeployments() {
  runtime=await loadDeployments({onRestore:restoreDeployment,onEdit:editDeployment});
  $('save-note').textContent=runtime?'Publication applies to new calls.':'Saved in this browser. Not connected to live calls.';
  updateControls();
}
async function editDeployment(record) {
  if(saving||publishing||loading)return;
  if(dirty){showError('Save your draft before loading the deployed settings.');return;}
  loading=true;updateControls();
  try {
    const encoded=record.image.src.split(',')[1],bytes=Uint8Array.from(atob(encoded),c=>c.charCodeAt(0));
    const blob=new Blob([bytes],{type:'image/png'}),decoded=await decode(blob);
    releaseImage();bitmap=decoded.img;sourceUrl=decoded.url;
    image={source:blob,preview:blob,name:'Driving image.png',crop:emptyCrop()};
    prompt.value=record.prompt;
    ranges.forEach(input=>input.value=String(image.crop[input.id]));
    displayImage(blob);imageStatus.textContent='Current driving image';showError('');changed();
  }catch {showError('Could not load the deployed settings.');}
  finally {loading=false;updateControls();}
}
function asDataUrl(blob) {
  return new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=()=>reject(new Error('Could not read the image'));reader.readAsDataURL(blob);});
}
async function publishRequest(payload) {
  const response=await fetch('./api/configuration',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const data=await response.json();
  if(!response.ok)throw new Error(data.detail||'Could not publish');
  return data;
}
async function restoreDeployment(id) {
  if(publishing||!runtime)return;
  publishing=true;showError('');updateControls();
  try {await publishRequest({restoreId:id,expectedRevision:runtime.revision});saveStatus.textContent='Deployment restored. Draft kept.';}
  catch(e){showError(e.message);}
  finally {publishing=false;await refreshDeployments();}
}
$('publish').addEventListener('click',async()=>{
  if(publishing||saving||loading||!runtime)return;
  publishing=true;showError('');updateControls();
  try {
    const imageData=image?await asDataUrl(image.preview):null;
    // Unchanged deployed image is kept byte-for-byte; prompt-only publication
    // does not re-encode the verified baseline or touch the renderer.
    await publishRequest({prompt:prompt.value,image:imageData===runtime.current.image.src?null:imageData,expectedRevision:runtime.revision});
    saveStatus.textContent='Published for new calls';
  }catch(e){showError(e.message);}
  finally {publishing=false;await refreshDeployments();}
});
$('refresh-deployments').addEventListener('click',()=>void refreshDeployments());
window.addEventListener('beforeunload' ,e=>{if(dirty||loading)e.preventDefault();});
void (async()=>{await loadSaved();await refreshDeployments();if(runtime&&!revision&&!dirty&&!image&&!prompt.value)await editDeployment(runtime.current);})();
