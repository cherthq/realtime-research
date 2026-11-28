import {checkRequest,json,runtimeSettings,privateHeaders} from '../lib/runtime-client.mjs';
const idPattern=/^[a-f0-9]{24}$/;
const files=new Set(['audio','caller','agent','video','transcript']);
export async function handleCalls(request,options={}) {
  if(request.method!=='GET')return json({detail:'Method not allowed'},405);
  const rejected=checkRequest(request);if(rejected)return rejected;
  const query=new URL(request.url).searchParams;
  const id=query.get('id'),file=query.get('download'),offset=query.get('offset')??'0';
  if((id&&!idPattern.test(id))||(file&&(!id||!files.has(file)))||!/^\d{1,7}$/.test(offset))return json({detail:'Invalid call request'},400);
  try {
    const settings=runtimeSettings(options.env),url=new URL(settings.url);
    url.searchParams.set('op',file?'ticket':id?'call':'calls');
    if(id)url.searchParams.set('id',id);
    if(file)url.searchParams.set('file',file);
    if(!id)url.searchParams.set('offset',offset);
    const response=await (options.fetch||fetch)(url,{headers:{Authorization:'Bearer '+settings.token},redirect:'error',signal:AbortSignal.timeout(20000)});
    const data=await response.json();
    if(!response.ok)return json(data,response.status);
    if(file){
      if(data.id!==id||data.file!==file||!/^\d+$/.test(data.expires)||! /^[a-f0-9]{64}$/.test(data.signature))throw Error('Invalid download ticket');
      const download=new URL(settings.url);
      download.search=new URLSearchParams({op:'download',id,file,expires:data.expires,signature:data.signature}).toString();
      return new Response(null,{status:302,headers:{...privateHeaders,Location:download.href}});
    }
    return json(data);
  }catch{return json({detail:'Call logs unavailable. Try again shortly.'},503);}
}
export default {fetch:handleCalls};
