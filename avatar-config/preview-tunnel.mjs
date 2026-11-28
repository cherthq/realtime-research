// Temporary loopback tunnel through the existing authenticated Jupyter terminal API.
// No files, packages, services, model weights, or configuration are changed remotely.
import net from 'node:net';
import {readFile,writeFile,rm} from 'node:fs/promises';
const config=JSON.parse(await readFile(new URL('./local/connection.json',import.meta.url)));
const base=new URL(config.base);
if(base.protocol!=='https:'||!base.hostname.endsWith('.lambdaspaces.com')||!config.token)throw new Error('Invalid workspace connection');
const headers={Authorization:`token ${config.token}`,'Content-Type':'application/json'};
const response=await fetch(new URL('/api/terminals',base),{method:'POST',headers,body:'{}',signal:AbortSignal.timeout(15000)});
if(!response.ok)throw new Error(`Workspace rejected connection (${response.status})`);
const terminal=(await response.json()).name;
const wsUrl=new URL(`/terminals/websocket/${encodeURIComponent(terminal)}`,base);wsUrl.protocol='wss:';wsUrl.searchParams.set('token',config.token);
const ws=new WebSocket(wsUrl);
const clients=new Map();let serial=0,buffer='',ready=false,stopping=false;
const server=net.createServer(socket=>{
  const id=String(++serial);clients.set(id,socket);
  send({op:'open',id});
  socket.on('data',bytes=>{for(let offset=0;offset<bytes.length;offset+=1024)send({op:'data',id,data:bytes.subarray(offset,offset+1024).toString('base64')});});
  socket.on('error',()=>socket.destroy());
  socket.on('close',()=>{clients.delete(id);send({op:'close',id});});
});
function send(message){if(ws.readyState===WebSocket.OPEN)ws.send(JSON.stringify(['stdin',JSON.stringify(message)+'\n']));}
async function stop(code=0){
  if(stopping)return;stopping=true;clearTimeout(startDeadline);
  for(const socket of clients.values())socket.destroy();server.close();
  ws.close();
  try{await fetch(new URL(`/api/terminals/${encodeURIComponent(terminal)}`,base),{method:'DELETE',headers,signal:AbortSignal.timeout(5000)});}catch{console.error('Could not close the temporary workspace terminal.');}
  await rm(new URL('./local/tunnel-state.json',import.meta.url),{force:true});process.exit(code);
}
const python=String.raw`import sys,socket,json,base64,threading
connections={}
lock=threading.Lock()
def emit(d):
 with lock:
  print('AVATAR_TUNNEL:'+json.dumps(d),flush=True)
def close(i):
 s=connections.pop(i,None)
 if s:
  try:s.shutdown(socket.SHUT_RDWR)
  except OSError:pass
  s.close()
def receive(i,s):
 try:
  while True:
   b=s.recv(16384)
   if not b:break
   emit({'op':'data','id':i,'data':base64.b64encode(b).decode()})
 except OSError:pass
 finally:
  close(i)
  emit({'op':'close','id':i})
emit({'op':'ready'})
for line in sys.stdin:
 try:
  d=json.loads(line);i=d['id']
  if d['op']=='open':
   s=socket.create_connection(('127.0.0.1',18837),timeout=5);s.settimeout(None);connections[i]=s
   threading.Thread(target=receive,args=(i,s),daemon=True).start()
  elif d['op']=='data' and i in connections:connections[i].sendall(base64.b64decode(d['data']))
  elif d['op']=='close':close(i)
 except Exception:
  if 'i' in locals():
   close(i);emit({'op':'close','id':i})
`;
ws.addEventListener('open',()=>{
  // Disable echo before sending any traffic. Python is executed from memory.
  const encoded=Buffer.from(python).toString('base64');
  ws.send(JSON.stringify(['stdin',`stty -echo; exec python3 -u -c "import base64;exec(base64.b64decode('${encoded}'))"\n`]));
});
ws.addEventListener('message',async event=>{
  let frame;try{frame=JSON.parse(event.data);}catch{return;}
  if(frame[0]!=='stdout')return;
  buffer+=frame[1];
  let newline;
  while((newline=buffer.indexOf('\n'))!==-1){
    const line=buffer.slice(0,newline).trim();buffer=buffer.slice(newline+1);
    const marker=line.indexOf('AVATAR_TUNNEL:');
    if(marker===-1)continue;
    let message;try{message=JSON.parse(line.slice(marker+14));}catch{continue;}
    if(message.op==='ready'&&!ready){
      ready=true;clearTimeout(startDeadline);
      server.listen(18837,'127.0.0.1',async()=>{
        await writeFile(new URL('./local/tunnel-state.json',import.meta.url),JSON.stringify({pid:process.pid,terminal,startedAt:new Date().toISOString()}),{mode:0o600});
        console.log('Existing Lambda web preview connected: http://127.0.0.1:18837');
      });
    }else if(message.op==='data'){clients.get(message.id)?.write(Buffer.from(message.data,'base64'));}
    else if(message.op==='close'){clients.get(message.id)?.end();}
  }
});
ws.addEventListener('error',()=>{console.error('Workspace tunnel connection failed.');void stop(1);});
ws.addEventListener('close',()=>{if(!stopping){console.error('Workspace tunnel disconnected.');void stop(1);}});
server.on('error',error=>{console.error(`Local tunnel unavailable: ${error.code}`);void stop(1);});
const startDeadline=setTimeout(()=>{console.error('Workspace tunnel startup timed out.');void stop(1);},15000);
process.on('SIGINT',()=>void stop());process.on('SIGTERM',()=>void stop());
