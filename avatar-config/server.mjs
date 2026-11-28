import {createServer} from 'node:http';
import {readFile} from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import {handleCalls} from './api/calls.js';
const routes=new Map([['/',['index.html','text/html']],['/index.html',['index.html','text/html']],['/calls.html',['calls.html','text/html']],['/calls.js',['calls.js','text/javascript']],['/calls.css',['calls.css','text/css']],['/app.js',['app.js','text/javascript']],['/model.js',['model.js','text/javascript']],['/deployments.js',['deployments.js','text/javascript']],['/deployments.json',['deployments.json','application/json']],['/style.css',['style.css','text/css']]]);
const headers={'Cache-Control':'no-store','X-Content-Type-Options':'nosniff',
  'Content-Security-Policy':"default-src 'self'; connect-src 'self'; img-src 'self' blob: data:; script-src 'self'; style-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"};
const send=(res,status,data)=>{res.writeHead(status,{...headers,'Content-Type':'application/json'});res.end(JSON.stringify(data));};

export function createDashboardServer({readConnection=()=>readFile(new URL('./local/runtime.json',import.meta.url),'utf8').then(JSON.parse),fetchRuntime=fetch}={}) {
  return createServer(async(req,res)=>{
    const origin=`http://127.0.0.1:${req.socket.localPort}`;
    // Binding loopback alone does not prevent DNS rebinding or cross-site writes.
    if(req.headers.host!==new URL(origin).host || (req.headers.origin&&req.headers.origin!==origin) || req.headers['sec-fetch-site']==='cross-site') {send(res,403,{detail:'Open the local dashboard directly'});return;}
    const pathname=new URL(req.url,origin).pathname;
    if(pathname==='/api/calls'){
      try{
        const connection=await readConnection();
        const workspace=JSON.parse(await readFile(new URL('./local/connection.json',import.meta.url),'utf8'));
        const response=await handleCalls(new Request(origin+req.url,{method:req.method,headers:req.headers}),{env:{AVATAR_RUNTIME_URL:workspace.base+'/api/avatar-config',AVATAR_RUNTIME_TOKEN:connection.token},fetch:fetchRuntime});
        res.writeHead(response.status,Object.fromEntries(response.headers));res.end(Buffer.from(await response.arrayBuffer()));
      }catch{send(res,503,{detail:'Call logs unavailable'});}
      return;
    }
    if(pathname==='/api/configuration') {
      if(!['GET','POST'].includes(req.method)){send(res,405,{detail:'Method not allowed'});return;}
      if(req.method==='POST'&&(req.headers.origin!==origin || !req.headers['content-type']?.startsWith('application/json'))) {send(res,403,{detail:'Use the local dashboard to publish'});return;}
      let connection;
      try {connection=await readConnection();}
      catch(error){send(res,error.code==='ENOENT'?404:503,{detail:'Runtime configuration is not connected'});return;}
      if(typeof connection.token!=='string'||connection.token.length<32){send(res,503,{detail:'Runtime configuration is not connected'});return;}
      try {
        let body;
        if(req.method==='POST'){
          const chunks=[];let length=0;
          for await(const chunk of req){length+=chunk.length;if(length>14*1024*1024){send(res,413,{detail:'Image is too large'});return;}chunks.push(chunk);}
          body=Buffer.concat(chunks);
        }
        const response=await fetchRuntime('http://127.0.0.1:18837/api/avatar-config',{
          method:req.method,headers:{Authorization:`Bearer ${connection.token}`,'Content-Type':'application/json'},body,
          redirect:'error',signal:AbortSignal.timeout(180000)
        });
        const data=await response.json();
        send(res,response.status,data);
      }catch {send(res,503,{detail:'Runtime unavailable. Reload deployments before retrying publication.'});}
      return;
    }
    const route=routes.get(pathname);
    if(!['GET','HEAD'].includes(req.method)){res.writeHead(405,headers);res.end();return;}
    if(!route){res.writeHead(404,headers);res.end('Not found');return;}
    try {
      let body;
      if(route[0]==='deployments.json') {
        try {body=await readFile(new URL('./local/deployments.json',import.meta.url));}
        catch(error){if(error.code!=='ENOENT')throw error;}
      }
      body??=await readFile(new URL(`./dist/${route[0]}`,import.meta.url));
      res.writeHead(200,{...headers,'Content-Type':`${route[1]}; charset=utf-8`});
      res.end(req.method==='HEAD'?undefined:body);
    }catch{res.writeHead(500,headers);res.end('Could not load preview');}
  });
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href) {
  createDashboardServer().listen(4198,'127.0.0.1',()=>console.log('Local: http://127.0.0.1:4198'));
}
