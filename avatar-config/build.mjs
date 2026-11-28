import {mkdir,readFile,writeFile,cp,rm} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {validateDeployments} from './dist/deployments.js';
import {runtimeRequest} from './lib/runtime-client.mjs';

// Only this allowlist becomes web assets. Runtime credentials and historical
// snapshots are never copied into the static output or browser JavaScript.
await rm('public',{recursive:true,force:true});await mkdir('public');
for(const name of ['index.html','style.css','app.js','model.js','deployments.js','calls.html','calls.js','calls.css'])await cp(`dist/${name}`,`public/${name}`);
const metadata=validateDeployments(JSON.parse(process.env.AVATAR_DEPLOYMENT_RECORDS));
const response=await runtimeRequest();if(!response.ok)throw Error('Could not verify existing deployment during build');
const live=await response.json();
const images=new Map();
for(const record of [live.current,...live.previous]){
  const raw=Buffer.from(record.image.src.split(',')[1],'base64');
  images.set(createHash('sha256').update(raw).digest('hex'),record.image.src);
}
for(const record of [metadata.current,...metadata.previous].filter(Boolean)){
  const src=images.get(record.image.sha256);
  if(!src)throw Error('Recorded driving image was not found in verified runtime history');
  record.image.src=src;delete record.image.sha256;
}
validateDeployments(metadata);
await mkdir('private',{recursive:true,mode:0o700});
await writeFile('private/recorded.json',JSON.stringify(metadata),{mode:0o600});
console.log('Built dashboard and verified private deployment records.');
