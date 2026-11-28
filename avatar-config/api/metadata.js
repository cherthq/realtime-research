import {readFile} from 'node:fs/promises';
import {checkRequest,json} from '../lib/runtime-client.mjs';
export default {
  async fetch(request) {
    if(request.method!=='GET')return json({detail:'Method not allowed'},405);
    const rejected=checkRequest(request);if(rejected)return rejected;
    try{return json(JSON.parse(await readFile(new URL('../private/recorded.json',import.meta.url),'utf8')));}
    catch(error){console.error('Deployment records unavailable',error.code??error.name);return json({detail:'Deployment records unavailable'},503);}
  }
};
