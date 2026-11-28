export function callLinks(data) {
  const preview=new URL(data.previewUrl);
  if(preview.protocol!=='http:' || !['127.0.0.1','localhost','[::1]'].includes(preview.hostname) || preview.username || preview.password) throw new Error('Invalid local preview URL');
  if(data.facetimeNumber!==null&&!/^\+[1-9]\d{7,14}$/.test(data.facetimeNumber)) throw new Error('Invalid FaceTime number');
  return {preview:preview.href,facetime:data.facetimeNumber?`facetime:${data.facetimeNumber}`:null};
}

export function validateDeployments(data) {
  callLinks(data);
  if(!Array.isArray(data.previous)||(data.current||data.previous.length||data.recordedAt!==null)&&!/^\d{4}-\d{2}-\d{2}$/.test(data.recordedAt)) throw new Error('Invalid deployment records');
  if(data.verifiedAt!==undefined&&!/^\d{4}-\d{2}-\d{2}$/.test(data.verifiedAt)) throw new Error('Invalid verification date');
  const ids=new Set();
  for(const record of [data.current,...data.previous].filter(Boolean)) {
    if(typeof record.id!=='string'||ids.has(record.id)||!/^\d{4}-\d{2}-\d{2}$/.test(record.date)||typeof record.detail!=='string'||typeof record.prompt!=='string'||!record.prompt.trim()||typeof record.greeting!=='string'||!record.image||typeof record.image.label!=='string'||typeof record.image.note!=='string') throw new Error('Invalid deployment record');
    // A future local export can include the actual image without external requests.
    if(record.image.src!==null && (typeof record.image.src!=='string'||!/^data:image\/(png|jpeg);base64,[A-Za-z0-9+/]+=*$/.test(record.image.src))) throw new Error('Invalid deployment image');
    ids.add(record.id);
  }
  return data;
}

function element(tag,className,text) {
  const node=document.createElement(tag);node.className=className;
  if(text!==undefined)node.textContent=text;
  return node;
}
function card(record,{restore,onRestore,onEdit}={}) {
  const article=element('article','deployment-card');
  const figure=element('figure','deployment-image');
  if(record.image.src){const img=element('img','');img.src=record.image.src;img.alt=record.image.label;figure.append(img);}
  else {const placeholder=element('div','missing-image');placeholder.append(element('span','image-symbol','▧'),element('span','',record.image.label),element('small','',record.image.note));figure.append(placeholder);}
  const content=element('div','deployment-content');
  const meta=element('div','deployment-meta');
  const date=element('time','',new Intl.DateTimeFormat('en-US',{month:'short',day:'numeric',year:'numeric',timeZone:'UTC'}).format(new Date(record.date)));date.dateTime=record.date;
  meta.append(date,element('span','',record.detail));
  const prompt=element('p','deployment-prompt',record.prompt);
  const toggle=element('button','text-button','Show full prompt');toggle.type='button';toggle.setAttribute('aria-expanded','false');
  prompt.id=`prompt-${record.id}`;toggle.setAttribute('aria-controls',prompt.id);
  toggle.addEventListener('click',()=>{const expanded=prompt.classList.toggle('expanded');toggle.textContent=expanded?'Show less':'Show full prompt';toggle.setAttribute('aria-expanded',String(expanded));});
  content.append(meta,element('h3','','Prompt'),prompt,toggle);
  if(record.greeting){const details=element('details','greeting');details.append(element('summary','','FaceTime greeting'),element('p','',record.greeting));content.append(details);}
  if(onEdit){const button=element("button","text-button","Edit");button.type="button";button.addEventListener("click",()=>onEdit(record));content.append(button);}
  if(restore){const button=element("button","action restore","Restore");button.type="button";button.addEventListener("click",()=>onRestore(record.id));content.append(button);}
  article.append(figure,content);return article;
}

export async function loadDeployments({onRestore,onEdit}={}) {
  const area=document.querySelector('.deployments'),status=document.getElementById('deployment-status');
  try {
    const response=await fetch('./deployments.json',{cache:'no-store'});
    if(!response.ok)throw new Error('Could not load deployment records');
    const data=validateDeployments(await response.json()),links=callLinks(data);
    const preview=document.getElementById('run-preview'),facetime=document.getElementById('call-facetime');
    preview.href=links.preview;preview.removeAttribute('aria-disabled');preview.title='Open the existing web call at '+links.preview;
    if(links.facetime){facetime.href=links.facetime;facetime.removeAttribute('aria-disabled');facetime.title=data.facetimeNumber;}
    else {facetime.removeAttribute('href');facetime.setAttribute('aria-disabled','true');facetime.title='No FaceTime number configured';}
    let runtime=null;
    try {
      const live=await fetch('./api/configuration',{cache:'no-store'});
      if(live.ok){
        runtime=await live.json();
        if(!Number.isSafeInteger(runtime.revision)||runtime.revision<1)throw new Error('Invalid revision');
        validateDeployments({...data,...runtime,recordedAt:runtime.current.date});
      }
    }catch {runtime=null;}
    const records=runtime||data;
    status.textContent=runtime?'Live · Applies to new calls':data.recordedAt?`${data.verifiedAt?'Verified':'Last recorded'} ${data.verifiedAt||data.recordedAt} · Not live-synced`:'No deployment connected';
    document.getElementById('current-deployment').replaceChildren(records.current?card(records.current,{onEdit:runtime?onEdit:null}):element('p','empty-deployments','No deployment recorded.'));
    document.getElementById('previous-deployments').replaceChildren(...(records.previous.length?records.previous.map(record=>card(record,{restore:Boolean(runtime&&onRestore),onRestore})):[element('p','empty-deployments','No previous deployments recorded.')]));
    return runtime;
  } catch {status.textContent='Deployment records unavailable. Reload to retry.';return null;}
  finally {area.setAttribute('aria-busy','false');}
}
