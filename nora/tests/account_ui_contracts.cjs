// Runs the real shipped inline JavaScript against a small DOM/fetch test double.
// This is NOT a browser rendering, accessibility or learned-model test.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../src/mental_health_pipeline/account_web_ui.py'),'utf8');
const script=source.match(/<script>([\s\S]*?)<\/script>/)[1];
class Element{
 constructor(id=''){this.id=id;this.children=[];this._value='';this.textContent='';this.hidden=false;this.checked=false;this.disabled=false}
 get value(){return this._value||(this.id==='chats'?this.children[0]?.value:'')||''}
 set value(x){this._value=x}
 append(x){this.children.push(x)}
 replaceChildren(...x){this.children=x;this.textContent='';if(this.id==='chats')this._value=''}
 scrollIntoView(){}
 set innerHTML(value){throw Error('Unsafe HTML interpolation')}
}
const ids=['auth','workspace','identity','memory','history','messages','notes','chats','message','password','alias','adult','notice','country','location','register','login','logout','save','new','send','refresh','forget','delete-chat','delete-account','export'];
const elements=Object.fromEntries(ids.map(id=>[id,new Element(id)]));
let expired=false;
const context={console,Date,Map,JSON,Error,encodeURIComponent,setTimeout,
 document:{getElementById:id=>elements[id],createElement:()=>new Element(),querySelectorAll:()=>Object.values(elements)},
 location:{reload(){}},confirm:()=>false,
 fetch:async(url,opts)=>{
  assert.equal(opts.credentials,'same-origin');
  let status=200,data;
  if(expired){status=401;data={detail:'sign_in_required'}}
  else if(url==='/accounts/me')data={alias:'fictional',memory_enabled:false,history_enabled:false};
  else if(url==='/accounts/chats')data=[{id:'chat-a',created:100},{id:'chat-b',created:101}];
  else if(url==='/accounts/memories')data=[];
  else if(url.endsWith('/messages'))data={response:'A checked reply <img src=x onerror=bad()>',route:'normal_chat',memory_notes_supplied:0,text_released:true};
  else if(url.startsWith('/accounts/chats/'))data=[];
  else throw Error('Unexpected request '+url);
  return {ok:status===200,status,json:async()=>data};
 }};
vm.createContext(context);vm.runInContext(script,context);
const tick=()=>new Promise(resolve=>setImmediate(resolve));
(async()=>{
 for(let i=0;i<5;i++)await tick();
 assert.equal(elements.workspace.hidden,false);
 elements.message.value='A fictional user message';
 await elements.send.onclick();
 assert.equal(elements.messages.children.length,2);
 assert.ok(elements.messages.children[1].textContent.includes('<img'));
 elements.chats.value='chat-b';await elements.chats.onchange();
 assert.equal(elements.messages.children.length,0);
 elements.chats.value='chat-a';await elements.chats.onchange();
 assert.equal(elements.messages.children.length,2,'history-off chat must remain visible in this tab');
 elements.password.value='not-a-real-password';elements.message.value='private unsent test draft';
 expired=true;await elements.refresh.onclick();
 assert.equal(elements.workspace.hidden,true,'expired auth must hide private UI');
 assert.equal(elements.messages.children.length,0);
 assert.equal(elements.notes.children.length,0);
 assert.equal(elements.password.value,'');assert.equal(elements.message.value,'');
 assert.equal(vm.runInContext('temporaryChats.size',context),0);
 context.testResponse={response:'Checked citation [source-1]',citations:[{source_id:'source-1',title:'<b>Untrusted title</b>',source:'Test source',url:'https://example.test'}]};
 vm.runInContext('showResponse(testResponse)',context);
 assert.ok(elements.messages.children.at(-1).textContent.includes('<b>Untrusted title</b>'));
 elements.messages.replaceChildren();context.testResponse.response='';
 vm.runInContext('showResponse(testResponse)',context);
 assert.equal(elements.messages.children.length,1,'withheld response must not expose source metadata');
 console.log(JSON.stringify({passed:true,contracts:['inline_script_executes','temporary_chat_switch','plaintext_response_rendering','expired_session_clears_private_view','approved_source_metadata_only'],browser_rendering_test:false}));
})().catch(error=>{console.error(error);process.exitCode=1});
