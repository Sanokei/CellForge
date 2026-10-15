// Input-only verification of the six frozen records. Never loads answers or predicts.
const fs=require('fs'),crypto=require('crypto');
const hash=s=>crypto.createHash('sha256').update(s).digest('hex');

function canon(x){if(Array.isArray(x))return '['+x.map(canon).join(',')+']';if(x&&typeof x==='object')return '{'+Object.keys(x).sort().map(k=>JSON.stringify(k)+':'+canon(x[k])).join(',')+'}';return JSON.stringify(x);}
function select(rows,exposure){
const family=n=>n.includes('.')?n.slice(0,n.lastIndexOf('.')):n,parent=new Map();
function find(a){if(!parent.has(a))parent.set(a,a);if(parent.get(a)!==a)parent.set(a,find(parent.get(a)));return parent.get(a);}
function union(a,b){a=find(a);b=find(b);if(a!==b)parent.set(a<b?b:a,a<b?a:b);}
for(const row of rows){const fs=[...new Set(row.function.map(f=>family(f.name)))].sort();fs.forEach(find);fs.slice(1).forEach(f=>union(fs[0],f));}
const exposed=new Set([...exposure.function_families,...exposure.additional_exposed_native_families,'triangle_properties','circle_properties'].filter(f=>parent.has(f)).map(find));
const names=new Set(exposure.function_names),duplicates=new Set(),ids=new Set(),eligible=[];
const norm=s=>s.replace(/\s+/g,' ').trim().toLowerCase();
for(const row of rows){
const publicInput={question:row.question,function:row.function},finger=hash(canon(publicInput));
const schemas=row.function.map(f=>({description:norm(f.description||''),parameters:f.parameters,original_name:f.name})).sort((a,b)=>canon({description:a.description,parameters:a.parameters})<canon({description:b.description,parameters:b.parameters})?-1:1);
let request=norm(row.question[0][0].content);schemas.forEach((f,i)=>{request=request.split(f.original_name.toLowerCase()).join('function'+i);});
const alias=hash(canon({request,functions:schemas.map(f=>({description:f.description,parameters:f.parameters}))})),bare=row.id.replace(/\.json$/,'').replace(/[\W_]/g,'').toLowerCase();
const fs=[...new Set(row.function.map(f=>family(f.name)))].sort(),component=find(fs[0]),q=row.question;
let reason;
if(q.length!==1||q[0].length!==1||q[0][0].role!=='user'||typeof q[0][0].content!=='string')reason='turn';
else if(row.function.length<2||row.function.length>4)reason='count';
else if(q[0][0].content.length>8192)reason='length';
else if(row.function.some(f=>!['dict','object'].includes(f.parameters.type)||Object.keys(f.parameters.properties||{}).length>6||Object.values(f.parameters.properties||{}).some(v=>!['string','integer','float','number','boolean'].includes(v.type))))reason='nonscalar';
else if(exposed.has(component)||row.id==='multiple_0'||row.function.some(f=>names.has(f.name)))reason='exposure';
else if(duplicates.has(alias)||ids.has(bare))reason='alias';
duplicates.add(alias);ids.add(bare);if(reason)continue;
eligible.push({id:row.id,component,input_sha256:finger,alias_sha256:alias,rank:hash('native-slots3-input-only-20261012:'+finger+':'+row.id)});
}
const groups=new Map();eligible.forEach(r=>{if(!groups.has(r.component))groups.set(r.component,[]);groups.get(r.component).push(r);});
const rank=(a,b)=>a.rank.localeCompare(b.rank);for(const g of groups.values())g.sort(rank);const ordered=[...groups.values()].sort((a,b)=>rank(a[0],b[0]));if(ordered.length<3)throw Error('HOLD: components');
const selected=ordered.slice(0,3).map(g=>g[0]),chosen=new Set(selected.map(r=>r.id));selected.push(...eligible.filter(r=>!chosen.has(r.id)).sort(rank).slice(0,3));
return {selected,eligible_count:eligible.length,component_count:groups.size};
}
const [source,exposurePath,freezePath]=process.argv.slice(2);
if(!source||!exposurePath||!freezePath)throw Error('Use pinned BFCL input JSONL, exposure JSON, SIX_INPUT_FREEZE.json');
const bytes=fs.readFileSync(source),rows=bytes.toString('utf8').split(/\r?\n/).filter(x=>x.trim()).map(JSON.parse),exposure=JSON.parse(fs.readFileSync(exposurePath,'utf8')),freeze=JSON.parse(fs.readFileSync(freezePath,'utf8'));
if(hash(bytes)!==freeze.source_sha256)throw Error('Pinned source byte hash differs');
if(hash(fs.readFileSync(exposurePath))!==freeze.exposure_manifest_sha256)throw Error('Exposure source byte hash differs');
const result=select(rows,exposure);if(result.selected.length!==6||result.selected.some((r,i)=>['id','component','input_sha256','alias_sha256','rank'].some(k=>r[k]!==freeze.selected[i][k])))throw Error('Frozen selection differs');
console.log(JSON.stringify({selection_verified:true,ids:result.selected.map(r=>r.id),eligible_count:result.eligible_count,component_count:result.component_count,prediction_calls:0}));
