import {readFile} from 'node:fs/promises';
import {join} from 'node:path';
import {checkRequest,json} from '../lib/runtime-client.mjs';
export default {
  async fetch(request) {
    if(request.method!=='GET')return json({detail:'Method not allowed'},405);
    const rejected=checkRequest(request);if(rejected)return rejected;
    try{return json(JSON.parse(await readFile(join(process.cwd(),'private','recorded.json'),'utf8')));}
    catch{console.error('Deployment records unavailable');return json({detail:'Deployment records unavailable'},503);}
  }
};
