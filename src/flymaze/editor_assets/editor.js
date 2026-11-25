"use strict";
const $ = id => document.getElementById(id);
const key = c => c.join(",");
const directions = [[1,0],[0,1],[0,-1],[-1,0]];
const token = new URLSearchParams(location.search).get("token") || "";
let state, tool="draw", view="edit", history=[], future=[], drawing=false, strokeTool, savedMaps=[];
const canvas=$("canvas"), ctx=canvas.getContext("2d");
let metrics={};
function template(kind) {
  let cells=[[1,2],[2,2],[3,2]],food;
  if(kind==="right"){cells.push([3,1],[3,0]);food=[3,0];}
  else {cells.push([3,3],[3,4]);food=[3,4];}
  if(kind==="s"){cells.push([4,4],[5,4],[6,4]);food=[6,4];}
  return {version:1,map_id:"custom_"+(kind==="s"?"s_turn":kind),width:8,height:6,cell_size_mm:10,start:[1,2],food,cells,timeout_seconds:kind==="s"?24:14};
}
function snapshot(){return JSON.stringify(state);}
function remember(){history.push(snapshot());if(history.length>100)history.shift();future=[];}
function message(text,error=false){$("status").textContent=text;$("status").classList.toggle("error",error);}
function readSettings(){state.map_id=$("mapid").value.trim();state.timeout_seconds=Number($("timeout").value);}
function sync(){for(const id of ["mapid","width","height","timeout"]){$(id).value=state[{mapid:"map_id",timeout:"timeout_seconds"}[id]||id];}render();}
function analyze(m){
  if(!m||m.version!==1)throw Error("不支援的地圖版本，需要 version: 1。");
  if(!/^custom_[a-z0-9_]{1,48}$/.test(m.map_id))throw Error("名稱須以 custom_ 開頭，後接 1–48 個小寫英文、數字或底線。");
  if(!Number.isInteger(m.width)||!Number.isInteger(m.height)||m.width<3||m.width>12||m.height<3||m.height>12)throw Error("寬高須為 3–12 格。");
  if(m.cell_size_mm!==10)throw Error("通道尺度固定為每格 10 mm。");
  if(!Number.isFinite(m.timeout_seconds)||m.timeout_seconds<1||m.timeout_seconds>300)throw Error("回合時間須為 1–300 秒。");
  const cellOK=c=>Array.isArray(c)&&c.length===2&&c.every(Number.isInteger)&&c[0]>=0&&c[0]<m.width&&c[1]>=0&&c[1]<m.height;
  if(!Array.isArray(m.cells)||m.cells.length<2||m.cells.length>m.width*m.height||!m.cells.every(cellOK)||!cellOK(m.start)||!cellOK(m.food))throw Error("通道、起點與食物須在畫布內，且至少有兩格通道。");
  const cells=new Set(m.cells.map(key));
  if(cells.size!==m.cells.length)throw Error("地圖含有重複通道格。");
  if(!cells.has(key(m.start))||!cells.has(key(m.food))||key(m.start)===key(m.food))throw Error("請在不同通道格上放置起點與食物。");
  if(!cells.has(key([m.start[0]+1,m.start[1]])))throw Error("起點右方須有通道，供果蠅朝 +X 出發。");
  for(const [x,y] of m.cells)if([[x+1,y],[x,y+1],[x+1,y+1]].every(c=>cells.has(key(c))))throw Error("請保持通道一格寬，避免 2 × 2 的開放房間。");
  const parents=new Map([[key(m.start),null]]),queue=[m.start];
  for(let i=0;i<queue.length;i++){const [x,y]=queue[i];for(const [dx,dy] of directions){const c=[x+dx,y+dy],k=key(c);if(cells.has(k)&&!parents.has(k)){parents.set(k,queue[i]);queue.push(c);}}}
  if(parents.size!==cells.size)throw Error("仍有不相連的通道。請用共用邊連接所有通道格。");
  const path=[];let c=m.food;while(c){path.push(c);c=parents.get(key(c));}path.reverse();return path;
}
function sizeCanvas(){const r=canvas.getBoundingClientRect(),d=devicePixelRatio||1;canvas.width=Math.round(r.width*d);canvas.height=Math.round(r.height*d);ctx.setTransform(d,0,0,d,0,0);return r;}
function render(){
  $("canvas-title").textContent=state.map_id;
  $("cell-count").innerHTML=state.cells.length+" <span>格</span>";
  $("undo").disabled=!history.length;$("redo").disabled=!future.length;
  let path=[];try{path=analyze(state);message("✓ 地圖檢查通過 · 通道相連、起點方向與尺度正確");}catch(e){message(e.message,true);}
  $("route-length").innerHTML=path.length?((path.length-1)*10)+" <span>mm</span>":"—";
  const r=sizeCanvas();ctx.clearRect(0,0,r.width,r.height);
  if(view==="iso"){drawIso(r,path);return;}
  const s=Math.min((r.width-50)/state.width,(r.height-55)/state.height),ox=(r.width-s*state.width)/2,oy=(r.height-s*state.height)/2-5;
  metrics={s,ox,oy};const cells=new Set(state.cells.map(key));
  for(let x=0;x<state.width;x++)for(let y=0;y<state.height;y++){
    const px=ox+x*s,py=oy+(state.height-1-y)*s;
    ctx.fillStyle=cells.has(key([x,y]))?"#fafcf4":"#dce4d7";ctx.fillRect(px,py,s,s);
    ctx.strokeStyle="#cbd6c5";ctx.lineWidth=.65;ctx.strokeRect(px,py,s,s);
    if(cells.has(key([x,y]))){ctx.strokeStyle="#668376";ctx.lineWidth=3;for(const [dx,dy] of directions){if(cells.has(key([x+dx,y+dy])))continue;ctx.beginPath();if(dx){const ex=px+(dx>0?s:0);ctx.moveTo(ex,py);ctx.lineTo(ex,py+s);}else{const ey=py+(dy>0?0:s);ctx.moveTo(px,ey);ctx.lineTo(px+s,ey);}ctx.stroke();}}
  }
  function center(c){return [ox+(c[0]+.5)*s,oy+(state.height-c[1]-.5)*s];}
  if(path.length){ctx.beginPath();path.forEach((c,i)=>{const [x,y]=center(c);if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y);});ctx.strokeStyle="#a1b77e";ctx.lineWidth=2;ctx.setLineDash([5,5]);ctx.stroke();ctx.setLineDash([]);}
  for(const [c,color,label] of [[state.start,"#176952","→"],[state.food,"#e98c4a","●"]]){
    const [x,y]=center(c);ctx.beginPath();ctx.arc(x,y,s*.25,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();ctx.fillStyle="#fff";ctx.textAlign="center";ctx.textBaseline="middle";ctx.font=`bold ${s*.27}px sans-serif`;ctx.fillText(label,x,y);
  }
  ctx.fillStyle="#83947d";ctx.font="10px Consolas,monospace";ctx.textAlign="center";ctx.textBaseline="middle";
  for(let x=0;x<state.width;x++)ctx.fillText(String((x-state.start[0])*10),ox+(x+.5)*s,oy+state.height*s+15);
  ctx.textAlign="right";for(let y=0;y<state.height;y++)ctx.fillText(String((y-state.start[1])*10),ox-7,oy+(state.height-y-.5)*s);
}
function drawIso(r,path){
  const scale=Math.min((r.width-65)/(state.width+state.height),(r.height-80)/((state.width+state.height)*.5+1));
  const ox=r.width/2+(state.height-state.width)*scale/2,oy=(r.height-(state.width+state.height)*scale*.5)/2+30;
  const p=(x,y,z=0)=>[ox+(x-y)*scale,oy+(x+y)*scale*.5-z*scale];
  function polygon(points,fill,line){ctx.beginPath();points.forEach((q,i)=>{if(i)ctx.lineTo(...q);else ctx.moveTo(...q);});ctx.closePath();ctx.fillStyle=fill;ctx.fill();if(line){ctx.strokeStyle=line;ctx.lineWidth=1;ctx.stroke();}}
  const cells=new Set(state.cells.map(key));
  const sorted=[...state.cells].sort((a,b)=>(a[0]+state.height-a[1])-(b[0]+state.height-b[1]));
  for(const [x,gy] of sorted){const y=state.height-1-gy;polygon([p(x,y),p(x+1,y),p(x+1,y+1),p(x,y+1)],"#f9fbef","#d3dcc7");}
  // Raised boundary faces use exactly the exposed edges of the 2D corridor cells.
  const edges=[];for(const [x,gy] of sorted){const y=state.height-1-gy;for(const [dx,dy] of directions){if(cells.has(key([x+dx,gy+dy])))continue;const a=dx?[x+(dx>0?1:0),y]:[x,y+(dy>0?0:1)];const b=dx?[a[0],y+1]:[x+1,a[1]];edges.push({a,b,depth:a[0]+a[1]+b[0]+b[1]});}}
  edges.sort((a,b)=>a.depth-b.depth);
  for(const {a,b} of edges)polygon([p(...a),p(...b),p(...b,.3),p(...a,.3)],a[0]===b[0]?"#608278":"#87a493","#43685c");
  for(const [cell,color,text] of [[state.start,"#176952","起點"],[state.food,"#e98c4a","食物"]]){const [x,y]=p(cell[0]+.5,state.height-cell[1]-.5,.12);ctx.beginPath();ctx.ellipse(x,y,scale*.17,scale*.11,0,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();ctx.font="11px sans-serif";ctx.textAlign="center";ctx.fillStyle=color;ctx.fillText(text,x,y-12);}
  $("hover").textContent="3D 外觀預覽 · 切回 2D 即可編輯";
}
function cellAt(event){const r=canvas.getBoundingClientRect(),{s,ox,oy}=metrics;const x=Math.floor((event.clientX-r.left-ox)/s),y=state.height-1-Math.floor((event.clientY-r.top-oy)/s);return x>=0&&y>=0&&x<state.width&&y<state.height?[x,y]:null;}
function paint(cell){if(!cell)return;const k=key(cell),set=new Set(state.cells.map(key));if(strokeTool==="erase"){if(k===key(state.start)||k===key(state.food))return;set.delete(k);}else{set.add(k);if(strokeTool==="start"&&k!==key(state.food))state.start=cell;if(strokeTool==="food"&&k!==key(state.start))state.food=cell;}state.cells=[...set].map(k=>k.split(",").map(Number));render();}
canvas.addEventListener("pointerdown",event=>{if(view!=="edit")return;const c=cellAt(event);if(!c)return;remember();drawing=true;strokeTool=event.button===2?"erase":tool;canvas.setPointerCapture(event.pointerId);paint(c);event.preventDefault();});
canvas.addEventListener("pointermove",event=>{if(view!=="edit")return;const c=cellAt(event);if(c)$("hover").textContent=`座標 (${(c[0]-state.start[0])*10}, ${(c[1]-state.start[1])*10}) mm`;if(drawing)paint(c);});
for(const e of ["pointerup","pointercancel","lostpointercapture"])canvas.addEventListener(e,()=>drawing=false);
canvas.addEventListener("contextmenu",e=>e.preventDefault());
function setTool(value){tool=value;document.querySelectorAll("[data-tool]").forEach(b=>b.classList.toggle("selected",b.dataset.tool===tool));if(view!=="edit")setView("edit");}
function setView(value){view=value;$("edit-view").classList.toggle("active",view==="edit");$("iso-view").classList.toggle("active",view==="iso");$("axis").hidden=view==="iso";$("hover").textContent="拖曳繪製通道 · 右鍵擦除";render();}
document.querySelectorAll("[data-tool]").forEach(b=>b.onclick=()=>setTool(b.dataset.tool));
document.querySelectorAll("[data-template]").forEach(b=>b.onclick=()=>{if(confirm("載入範本會替換目前畫布，是否繼續？")){remember();state=template(b.dataset.template);sync();}});
$("edit-view").onclick=()=>setView("edit");$("iso-view").onclick=()=>setView("iso");
$("undo").onclick=()=>{if(history.length){future.push(snapshot());state=JSON.parse(history.pop());sync();}};
$("redo").onclick=()=>{if(future.length){history.push(snapshot());state=JSON.parse(future.pop());sync();}};
for(const id of ["mapid","timeout"])$(id).addEventListener("change",()=>{remember();readSettings();render();});
$("resize").onclick=()=>{const w=Number($("width").value),h=Number($("height").value);if(!Number.isInteger(w)||!Number.isInteger(h)||w<3||w>12||h<3||h>12){message("寬高須為 3–12 格。",true);return;}if(confirm("套用尺寸會清空目前通道，是否繼續？")){remember();state.width=w;state.height=h;state.start=[0,1];state.food=[2,1];state.cells=[[0,1],[1,1],[2,1]];sync();}};
async function api(path,payload){const response=await fetch(path,{method:payload?"POST":"GET",headers:{"X-FlyMaze-Token":token,"Content-Type":"application/json"},body:payload?JSON.stringify(payload):undefined});const result=await response.json();if(!response.ok){const error=new Error(result.error||"請求失敗");error.status=response.status;throw error;}return result;}
async function loadList(){try{const result=await api("/api/maps");savedMaps=result.maps;$("saved").replaceChildren(new Option("選擇地圖…",""),...savedMaps.map(m=>new Option(m.map_id,m.map_id)));if(result.errors.length)message(result.errors.join("；"),true);}catch(e){message("無法讀取清單："+e.message,true);}}
$("reload").onclick=loadList;
$("saved").onchange=()=>{const found=savedMaps.find(m=>m.map_id===$("saved").value);if(found&&confirm("載入已儲存地圖會替換目前畫布，是否繼續？")){remember();state=structuredClone(found);sync();}};
$("save").onclick=async()=>{readSettings();try{analyze(state);$("save").disabled=true;let result;try{result=await api("/api/save",{map:state});}catch(e){if(e.status===409&&confirm("已有同名地圖，是否覆寫？"))result=await api("/api/save",{map:state,overwrite:true});else throw e;}await loadList();message("✓ 已儲存 "+result.map_id+" · "+result.route_length_mm+" mm · "+result.wall_count+" 面牆");}catch(e){message(e.message,true);}finally{$("save").disabled=false;}};
$("export").onclick=()=>{readSettings();try{analyze(state);const url=URL.createObjectURL(new Blob([JSON.stringify(state,null,2)+"\n"],{type:"application/json"})),a=document.createElement("a");a.href=url;a.download=state.map_id+".json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);message("✓ 已匯出 "+state.map_id+".json");}catch(e){message(e.message,true);}};
$("import").onclick=()=>$("file").click();$("file").onchange=async()=>{try{const file=$("file").files[0];if(!file)return;if(file.size>100000)throw Error("檔案大小須小於 100 KB。");const m=JSON.parse((await file.text()).replace(/^\uFEFF/,""));analyze(m);remember();state=m;sync();}catch(e){message("匯入失敗："+e.message,true);}finally{$("file").value="";}};
document.addEventListener("keydown",e=>{if(["INPUT","SELECT","TEXTAREA"].includes(e.target.tagName))return;if(e.key>="1"&&e.key<="4")setTool(["draw","erase","start","food"][Number(e.key)-1]);if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==="z"){e.preventDefault();$(e.shiftKey?"redo":"undo").click();}});
window.addEventListener("resize",()=>render());
state=template("left");sync();loadList();
