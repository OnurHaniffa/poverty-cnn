const puppeteer=require('puppeteer-core'); const path=require('path');
const CHROME='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
(async()=>{const files=process.argv.slice(2);
const b=await puppeteer.launch({executablePath:CHROME,headless:'new',args:['--force-device-scale-factor=1','--no-sandbox']});
const p=await b.newPage(); await p.setViewport({width:1920,height:1080}); let any=false;
for(const f of files){await p.goto('file://'+path.resolve(f),{waitUntil:'networkidle0'}); await p.evaluate(async()=>{await document.fonts.ready;});
const iss=await p.evaluate(()=>{const W=1920,H=1080,out=[];
const ov=(a,b)=>{const ix=Math.max(0,Math.min(a.right,b.right)-Math.max(a.left,b.left));const iy=Math.max(0,Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top));return Math.max(0,ix)*Math.max(0,iy);};
const els=[...document.querySelectorAll('body *')].filter(e=>{const s=getComputedStyle(e);if(s.position!=='absolute')return false;if(e.tagName.toLowerCase()==='svg')return false;const c=(typeof e.className==='string')?e.className:'';if(/(^|\s)(grain|gl|svg)(\s|$)/.test(c))return false;if(!c.trim())return false;const r=e.getBoundingClientRect();if(r.width<3||r.height<3)return false;if(r.width*r.height>0.55*W*H)return false;if(r.height>850)return false;return true;});
for(const e of els){const r=e.getBoundingClientRect();if(r.left<-2||r.top<-2||r.right>W+2||r.bottom>H+3)out.push(`OFF-CANVAS .${e.className}`);}
for(let i=0;i<els.length;i++)for(let j=i+1;j<els.length;j++){const a=els[i],b=els[j];if(a.contains(b)||b.contains(a))continue;if(ov(a.getBoundingClientRect(),b.getBoundingClientRect())>500)out.push(`OVERLAP .${a.className} ✕ .${b.className}`);}
const T=[...document.querySelectorAll('svg text')].map(e=>({t:e.textContent.trim().slice(0,20),r:e.getBoundingClientRect()})).filter(o=>o.r.width>1);
for(let i=0;i<T.length;i++)for(let j=i+1;j<T.length;j++){if(ov(T[i].r,T[j].r)>40)out.push(`TEXT-OVERLAP "${T[i].t}" ✕ "${T[j].t}"`);}
// NEW: text overflow (content wider/taller than its box)
[...document.querySelectorAll('body *')].forEach(e=>{if(e.children.length!==0||e.namespaceURI&&e.namespaceURI.includes('svg'))return;const fs=parseFloat(getComputedStyle(e).fontSize)||0;if(fs>50)return;const t=(e.textContent||'').trim();if(!t)return;if(e.scrollWidth>e.clientWidth+14||e.scrollHeight>e.clientHeight+10){const c=(typeof e.className==='string')?e.className:'';out.push(`OVERFLOW .${c||e.tagName} "${t.slice(0,18)}"`);}});
// NEW: content sitting on the globe (orbit/core) — text whose box overlaps the globe svg ink area on the right
const g=document.querySelector('.gl, .svg'); if(g){const gb={left:980,top:120,right:1860,bottom:980}; // globe ink region
 for(const e of els){const c=(typeof e.className==='string')?e.className:'';if(/sig|signum|t |^t$|gl/.test(c))continue;const r=e.getBoundingClientRect();if(r.left>700 && ov(r,gb)>6000 && (e.textContent||'').trim())out.push(`ON-GLOBE .${c}`);}}
return out;});
if(iss.length){any=true;console.log(`\n### ${f.split('/').pop()} — ${iss.length}`);iss.forEach(x=>console.log('   '+x));}else console.log(`OK   ${f.split('/').pop()}`);}
await b.close();process.exit(any?1:0);})();
