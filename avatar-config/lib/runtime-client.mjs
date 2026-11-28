export const MAX_BODY=3*1024*1024;
export const privateHeaders={'Cache-Control':'private, no-store','X-Content-Type-Options':'nosniff','Referrer-Policy':'no-referrer'};
export const json=(body,status=200)=>Response.json(body,{status,headers:privateHeaders});

export function checkRequest(request) {
  if(!['GET','POST'].includes(request.method))return json({detail:'Method not allowed'},405);
  const origin=request.headers.get('origin');
  if(request.headers.get('sec-fetch-site')==='cross-site'||(origin&&origin!==new URL(request.url).origin))return json({detail:'Cross-origin access denied'},403);
  if(request.method==='POST'&&(origin!==new URL(request.url).origin||!request.headers.get('content-type')?.startsWith('application/json')))return json({detail:'Publish from the dashboard'},403);
  return null;
}
export function runtimeSettings(env=process.env) {
  const url=new URL(env.AVATAR_RUNTIME_URL);
  if(url.protocol!=='https:'||url.username||url.password||url.search||url.hash||url.pathname!=='/api/avatar-config')throw Error('Invalid runtime configuration');
  if(typeof env.AVATAR_RUNTIME_TOKEN!=='string'||env.AVATAR_RUNTIME_TOKEN.length<32)throw Error('Missing runtime authorization');
  return {url:url.href,token:env.AVATAR_RUNTIME_TOKEN};
}
export async function runtimeRequest(method='GET',body,options={}) {
  const {url,token}=runtimeSettings(options.env);
  return (options.fetch||fetch)(url,{method,body,headers:{Authorization:`Bearer ${token}`,'Content-Type':'application/json'},redirect:'error',signal:AbortSignal.timeout(180000)});
}
export async function readBody(request) {
  if(Number(request.headers.get('content-length'))>MAX_BODY)throw Error('Body too large');
  const reader=request.body?.getReader();if(!reader)return '';
  const chunks=[];let size=0;
  while(true){const {done,value}=await reader.read();if(done)break;size+=value.byteLength;if(size>MAX_BODY){await reader.cancel();throw Error('Body too large');}chunks.push(Buffer.from(value));}
  return Buffer.concat(chunks).toString('utf8');
}
