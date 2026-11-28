const $=id=>document.getElementById(id);
let next=null,loading=false;
const element=(tag,text,className)=>{const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(className)node.className=className;return node;};
const duration=seconds=>{const total=Math.max(0,Math.round(seconds||0));return total>=3600?`${Math.floor(total/3600)}h ${Math.floor(total%3600/60)}m`:`${Math.floor(total/60)}m ${total%60}s`;};
const labels={audio:'Audio · both speakers',video:'Video + audio',caller:'Caller audio',agent:'Agent audio',transcript:'Transcript .txt'};
async function read(query='') {const r=await fetch('/api/calls'+query,{cache:'no-store'});if(!r.ok)throw Error('Call logs unavailable. Try again shortly.');return r.json();}
async function transcript(call){
  $('transcript-body').replaceChildren(element('p','Loading transcript…'));$('transcript').showModal();
  try{const data=await read('?id='+encodeURIComponent(call.id));const rows=data.transcript||[];
    $('transcript-body').replaceChildren(...rows.filter(row=>row.text).map(row=>{const item=element('div',undefined,'utterance');item.append(element('strong',`${row.speaker==='assistant'?'Agent':'Caller'} · ${duration(row.seconds)}`),element('p',row.text));return item;}));
    if(!rows.some(row=>row.text))$('transcript-body').append(element('p','No speech transcript was captured.'));
  }catch(error){$('transcript-body').replaceChildren(element('p',error.message));}
}
function card(call){
  const article=element('article',undefined,'call-card'),heading=element('div',undefined,'call-heading'),title=element('div');
  title.append(element('h2',new Date(call.startedAt).toLocaleString([], {dateStyle:'medium',timeStyle:'short'})),element('div',`${call.source==='facetime'?'FaceTime':'Web preview'} · ${duration(call.duration)}`,'call-meta'));
  heading.append(title,element('span',({recording:'Recording',processing:'Preparing downloads',complete:'Recorded',partial:'Partial recording',failed:'Recording failed',interrupted:'Interrupted',no_media:'No connected media'})[call.status]||call.status,'call-state'));article.append(heading);
  if(call.error)article.append(element('p',call.error,'call-error'));
  const downloads=element('div',undefined,'call-files');
  const view=element('button','View transcript');view.type='button';view.addEventListener('click',()=>transcript(call));downloads.append(view);
  for(const [kind,label] of Object.entries(labels))if(call.files?.[kind]){const link=element('a',label);link.href='/api/calls?'+new URLSearchParams({id:call.id,download:kind});downloads.append(link);}
  article.append(downloads);return article;
}
async function load(more=false){
  if(loading)return;loading=true;$('refresh').disabled=true;$('more').disabled=true;
  try{const data=await read(more?'?offset='+next:'');if(!more)$('calls').replaceChildren();
    $('calls').append(...data.calls.map(card));next=data.nextOffset;$('more').hidden=next===null;
    $('status').textContent=data.storageFreeBytes<data.storageReserveBytes?'Recording storage is low. Free space is required to continue recording.':data.historyNote;
    if(!more&&!data.calls.length){const empty=element('section',undefined,'empty-calls');empty.append(element('h2','No calls recorded yet'),element('p','Your next FaceTime or web call will appear here.'));$('calls').append(empty);}
  }catch(error){$('status').textContent=error.message;}finally{loading=false;$('refresh').disabled=false;$('more').disabled=false;}
}
$('refresh').addEventListener('click',()=>load());$('more').addEventListener('click',()=>load(true));$('close-transcript').addEventListener('click',()=>$('transcript').close());
load();
