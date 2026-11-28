import {checkRequest,json,readBody,runtimeRequest} from '../lib/runtime-client.mjs';
export default {
  async fetch(request) {
    const rejected=checkRequest(request);if(rejected)return rejected;
    let body;
    if(request.method==='POST'){
      try{body=await readBody(request);JSON.parse(body);}
      catch{return json({detail:'Invalid or oversized deployment request'},413);}
    }
    try {
      const response=await runtimeRequest(request.method,body);
      const data=await response.json();
      // Log status only: no prompts, images, tokens or upstream URLs.
      if(response.status>=500)console.error('Avatar runtime unavailable',response.status);
      return json(data,response.status);
    }catch{
      console.error('Avatar runtime connection failed');
      return json({detail:'Runtime unavailable. Refresh deployments before retrying.'},503);
    }
  }
};
