import {createServer} from 'node:http';
import {readFile} from 'node:fs/promises';
const routes=new Map([['/',['index.html','text/html']],['/index.html',['index.html','text/html']],['/app.js',['app.js','text/javascript']],['/model.js',['model.js','text/javascript']],['/deployments.js',['deployments.js','text/javascript']],['/deployments.json',['deployments.json','application/json']],['/style.css',['style.css','text/css']]]);
const server=createServer(async(req,res)=>{
  const route=routes.get(new URL(req.url,'http://127.0.0.1').pathname);
  if(!['GET','HEAD'].includes(req.method)){res.writeHead(405);res.end();return;}
  if(!route){res.writeHead(404);res.end('Not found');return;}
  try {
    let body;
    if(route[0]==='deployments.json') {
      try {body=await readFile(new URL('./local/deployments.json',import.meta.url));}
      catch(error){if(error.code!=='ENOENT')throw error;}
    }
    body??=await readFile(new URL(`./dist/${route[0]}`,import.meta.url));
    res.writeHead(200,{'Content-Type':`${route[1]}; charset=utf-8`,'Cache-Control':'no-store','X-Content-Type-Options':'nosniff',
      'Content-Security-Policy':"default-src 'self'; connect-src 'self'; img-src 'self' blob: data:; script-src 'self'; style-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"});
    res.end(req.method==='HEAD'?undefined:body);
  }catch{res.writeHead(500);res.end('Could not load preview');}
});
server.listen(4198,'127.0.0.1',()=>console.log('Local: http://127.0.0.1:4198'));
