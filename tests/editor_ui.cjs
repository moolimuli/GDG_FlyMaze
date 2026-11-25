/* Run with node tests/editor_ui.cjs. No browser or npm dependencies required.
 * This checks UI state and event handlers with a canvas/DOM test double;
 * it does not claim pixel-level browser rendering coverage. */
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
class Element {
  constructor(id){this.id=id;this.value='';this.textContent='';this.innerHTML='';this.disabled=false;this.listeners={};this.classList={toggle(){}};this.dataset={};}
  addEventListener(name,fn){this.listeners[name]=fn;}
  replaceChildren(...children){this.children=children;}
  getBoundingClientRect(){return {left:0,top:0,width:640,height:400};}
  getContext(){return new Proxy({}, {get:(target,name)=>target[name]||(()=>{}),set:(target,name,value)=>(target[name]=value,true)});}
  setPointerCapture(){}
  click(){return this.onclick?.();}
}
const elements=new Map(),get=id=>{if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id);};
const toolButtons=['draw','erase','start','food'].map(tool=>{const e=new Element(tool);e.dataset.tool=tool;return e;});
const templateButtons=['left','right','s'].map(kind=>{const e=new Element(kind);e.dataset.template=kind;return e;});
const context=vm.createContext({
  document:{getElementById:get,querySelectorAll:s=>s==='[data-tool]'?toolButtons:templateButtons,addEventListener(){},createElement:()=>new Element('a')},
  window:{addEventListener(){}},location:{search:'?token=test'},devicePixelRatio:1,
  URLSearchParams,URL,Blob,Set,Map,structuredClone,console,confirm:()=>true,
  Option:class {constructor(text,value){this.text=text;this.value=value;}},setTimeout,
  fetch:async()=>({ok:true,json:async()=>({maps:[],errors:[]})}),
});
vm.runInContext(fs.readFileSync('src/flymaze/editor_assets/editor.js','utf8'),context);
const run=code=>vm.runInContext(code,context);
assert.match(get('route-length').innerHTML,/40/);
for(const kind of ['left','right','s'])assert.equal(run(`analyze(template('${kind}')).length>1`),true);
run("setView('iso')");assert.equal(get('axis').hidden,true);
run("setView('edit')");assert.equal(get('axis').hidden,false);
const before=run('snapshot()');
run("remember();strokeTool='draw';paint([2,1])");assert.equal(run('state.cells.length'),6);
get('undo').click();assert.equal(run('snapshot()'),before);
get('redo').click();assert.equal(run('state.cells.length'),6);
get('undo').click();
assert.throws(()=>run("analyze({...state,map_id:'test_straight_unknown'})"));
assert.throws(()=>run("analyze({...state,cells:[...state.cells,[7,5]]})"));
assert.throws(()=>run("analyze({...state,cells:[...state.cells,[2,3]]})"));
run("setTool('food');strokeTool='food';paint([3,3])");
assert.equal(run('key(state.food)'),'3,3');
assert.match(get('route-length').innerHTML,/30/);
console.log('PASS: initial render, 3 templates, 2D/3D, paint, undo/redo, validation, food placement');
