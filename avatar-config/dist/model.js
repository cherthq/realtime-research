/** @typedef {{x:number,y:number,zoom:number}} Crop */
/** @typedef {{source:Blob, preview:Blob, name:string, crop:Crop}} DrivingImage */
/** @typedef {{prompt:string,image:DrivingImage|null,revision:number}} Configuration */
export const emptyCrop = () => ({x:50,y:50,zoom:1});
export const MAX_BYTES = 10 * 1024 * 1024;
export const MAX_PIXELS = 24_000_000;

export function validatePrompt(prompt) {
  if (!prompt.trim()) throw new Error('Enter a prompt first.');
  if (prompt.length > 12000) throw new Error('Keep the prompt under 12,000 characters.');
}

/** Returns the bounded 4:3 source rectangle, independent of display size. */
export function cropRectangle(width,height,crop) {
  const w = Math.min(width,height*4/3) / Math.max(1,Math.min(3,crop.zoom));
  const h = w*3/4;
  return {x:(width-w)*Math.max(0,Math.min(100,crop.x))/100,
    y:(height-h)*Math.max(0,Math.min(100,crop.y))/100,width:w,height:h};
}

/** Reject unsupported and oversized inputs before browser decoding where possible. */
export async function validateImage(file) {
  if (!['image/jpeg','image/png'].includes(file.type)) throw new Error('Choose a PNG or JPG image.');
  if (!file.size || file.size > MAX_BYTES) throw new Error('Choose an image under 10 MB.');
  const bytes = new Uint8Array(await file.arrayBuffer());
  const view = new DataView(bytes.buffer);
  if (file.type === 'image/png') {
    if (bytes.length < 33 || ![137,80,78,71,13,10,26,10].every((v,i)=>v===bytes[i])) throw new Error('This PNG is damaged. Try another image.');
    checkDimensions(view.getUint32(16),view.getUint32(20));
    for (let i=8;i+12<=bytes.length;) {
      const length=view.getUint32(i);
      if (i+12+length>bytes.length) throw new Error('This PNG is incomplete. Try another image.');
      if (String.fromCharCode(...bytes.slice(i+4,i+8))==='acTL') throw new Error('Choose a still image, not an animated PNG.');
      i+=12+length;
    }
  } else {
    if (bytes[0]!==255 || bytes[1]!==216 || bytes[2]!==255) throw new Error('This JPG is damaged. Try another image.');
    // Read SOF dimensions before allocating a decoded image.
    for (let i=2;i+4<=bytes.length;) {
      if (bytes[i]!==255) break;
      const marker=bytes[i+1];
      if (marker===255) {i++;continue;}
      if (marker===217 || marker===218) break;
      const length=view.getUint16(i+2);
      if (length<2 || i+2+length>bytes.length) throw new Error('This JPG is incomplete. Try another image.');
      if ([192,193,194,195,197,198,199,201,202,203,205,206,207].includes(marker)) {
        if (length<7) throw new Error('This JPG is damaged. Try another image.');
        checkDimensions(view.getUint16(i+7),view.getUint16(i+5)); break;
      }
      i+=length+2;
    }
  }
}
export function checkDimensions(width,height) {
  if (!width || !height || width*height>MAX_PIXELS) throw new Error('Choose an image with at most 24 million pixels.');
}
export function assertRevision(current,expected) {
  if (current!==expected) throw new Error('Settings changed in another tab. Reload the saved version before saving.');
}

/** @returns {Promise<Configuration>} */
export function loadConfiguration(db) {
  return new Promise((resolve,reject)=>{
    const tx=db.transaction('settings','readonly');
    const request=tx.objectStore('settings').get('avatar');
    request.onsuccess=()=>resolve(request.result ?? {prompt:'',image:null,revision:0});
    request.onerror=()=>reject(new Error('Could not load local settings. Reload to try again.'));
  });
}
/** Atomic local save; a future remote adapter must supply its own authorization. */
export function saveConfiguration(db,configuration,expected) {
  validatePrompt(configuration.prompt);
  return new Promise((resolve,reject)=>{
    const tx=db.transaction('settings','readwrite');
    const store=tx.objectStore('settings');
    let error;
    let result;
    const read=store.get('avatar');
    read.onsuccess=()=>{
      try {
        assertRevision(read.result?.revision ?? 0,expected);
        result={...configuration,revision:expected+1};
        store.put(result,'avatar');
      } catch(e) {error=e;tx.abort();}
    };
    tx.oncomplete=()=>resolve(result);
    tx.onabort=tx.onerror=()=>reject(error ?? new Error('Could not save in this browser. Free some local storage and try again.'));
  });
}
export function openLocalStore() {
  return new Promise((resolve,reject)=>{
    const request=indexedDB.open('standalone-avatar-config',1);
    request.onupgradeneeded=()=>request.result.createObjectStore('settings');
    request.onsuccess=()=>resolve(request.result);
    request.onerror=()=>reject(new Error('Local storage is unavailable. Enable browser storage and reload.'));
    request.onblocked=()=>reject(new Error('Close other avatar settings tabs, then reload.'));
  });
}
