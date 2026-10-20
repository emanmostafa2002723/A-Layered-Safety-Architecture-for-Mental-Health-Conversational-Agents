// Executes the shipped view and connection code, with DOM/fetch doubles.
// Not a browser rendering or learned-model test.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const root=path.join(__dirname,'../src/mental_health_pipeline/web_nora');
class Element{
 constructor(){this.dataset={};this.value='';this.textContent='';this.innerHTML='';this.open=false;this.checked=false;this.disabled=false;this.handlers={};this.classList={add(){},remove(){},toggle(){}};}
 querySelectorAll(){return [];}
 querySelector(){return new Element();}
 focus(){} click(){} scrollIntoView(){}
 replaceChildren(){this.innerHTML='';}
 addEventListener(name,fn){this.handlers[name]=fn;}
 showModal(){this.open=true;}
 close(){this.open=false;}
}
const elements=new Map();
const el=s=>{if(!elements.has(s))elements.set(s,new Element());return elements.get(s);};
const events={};
const calls=[];
let expired=false,withhold=false,wrongPassword=false,failSettings=false,holdReply=null;
let nextChat=0;
let preferences={display_name:'',avatar:{character:2,colour:0,accessory:'none'},theme:'day',text:'default',motion:'auto',quiet:false,country:null,location_confirmed:false,bookmarks:[]};
let consent={memory_enabled:false,history_enabled:false};
const context={console,Date,Map,JSON,Error,TextEncoder,URL,URLSearchParams,Blob,
 crypto:{randomUUID:()=> 'test-id'},setTimeout:()=>1,clearTimeout(){},setInterval:()=>1,clearInterval(){},
 requestAnimationFrame:fn=>fn(),
 document:{querySelector:el,querySelectorAll:()=>[],createElement:()=>new Element(),
  documentElement:{dataset:{}},body:new Element(),addEventListener:(name,fn)=>events[name]=fn},
 window:{scrollTo(){},addEventListener(){}},
 fetch:async(url,options)=>{
  assert.equal(options.credentials,'same-origin');assert.equal(options.cache,'no-store');
  const method=options.method,body=options.body?JSON.parse(options.body):undefined;
  calls.push({url,method,body});
  if(method!=='GET'){assert.equal(options.headers['X-Memory-Client'],'1');assert.equal(options.headers['Content-Type'],'application/json');}
  let status=200,data;
  if(expired){status=401;data={detail:'sign_in_required'};}
  else if(url==='/accounts/login'&&wrongPassword){status=401;data={detail:'invalid_credentials'};}
  else if(url==='/accounts/me')data={id:'owner-a',alias:'synthetic',...consent};
  else if(url==='/accounts/interface-preferences'){
   if(failSettings&&method==='PUT'){status=503;data={detail:'service_unavailable'};}
   else {if(method==='PUT')preferences=body;data=structuredClone(preferences);}
  }
  else if(url==='/accounts/preferences'){consent=body;data={...body};}
  else if(url==='/accounts/chats'&&method==='POST')data={id:'chat-'+(++nextChat),created_at:1800000000,expires_at:1802592000};
  else if(url==='/accounts/chats')data=Array.from({length:nextChat},(_,i)=>({id:'chat-'+(i+1),created:1800000000,expires:1802592000,turn:1}));
  else if(url==='/accounts/memories')data=[];
  else if(url.endsWith('/messages')){
   if(holdReply)await new Promise(resolve=>{holdReply.resolve=resolve;});
   data={text_released:!withhold,response:withhold?'UNCHECKED SECRET':'A checked reply <img src=x onerror=bad()>',
    citations:[{source_id:'s1',title:'Untrusted <script> title',url:'javascript:bad()'}]};
  }
  else if(url.startsWith('/accounts/chats/'))data=[];
  else if(url==='/accounts/logout')data={signed_out:true};
  else throw Error('Unexpected request '+url);
  return {ok:status===200,status,json:async()=>data};
 }};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(root,'app.js'),'utf8'),context);
vm.runInContext(fs.readFileSync(path.join(root,'connection.js'),'utf8'),context);
const run=s=>vm.runInContext(s,context);
const tick=()=>new Promise(resolve=>setImmediate(resolve));
(async()=>{
 for(let i=0;i<8;i++)await tick();
 assert.equal(run('S.signedIn'),true);
 assert.equal(run('S.memory'),false);assert.equal(run('S.history'),false);
 await run('send("A fictional message with a typo.")');
 assert.equal(calls.find(c=>c.url.endsWith('/messages')).body.message,'A fictional message with a typo.');
 assert.equal(run('S.current.messages.length'),2);
 assert.ok(run('messageHTML(S.current.messages[1])').includes('&lt;img'));
 assert.ok(!run('messageHTML(S.current.messages[1])').includes('href="javascript:'));
 assert.ok(run('messageHTML(S.current.messages[1])').includes('Untrusted &lt;script&gt;'));
 const first=run('S.current.id');
 await run('freshChat()');await run('openChat("'+first+'")');
 assert.equal(run('S.current.messages.length'),2,'unsaved tab context survives switching');
 withhold=true;await run('send("A different fictional message.")');
 assert.ok(!run('chatView()').includes('UNCHECKED SECRET'));
 assert.equal(run('S.current.messages.at(-1).released'),false);
 withhold=false;
 assert.ok(!run('messageHTML(displayedResponse({text_released:false,response:"secret",citations:[{title:"secret source"}]}))').includes('secret'));
 await run('saveUI({display_name:"<b>Fictional</b>",avatar:{character:5,colour:1,accessory:"star"}})');
 assert.ok(run('homeView()').includes('&lt;b&gt;Fictional&lt;/b&gt;'));
 assert.equal(preferences.avatar.character,5);
 failSettings=true;
 await assert.rejects(run('saveUI({theme:"evening"})'));
 assert.equal(run('S.theme'),'day','failed persistence cannot pretend to succeed');
 failSettings=false;
 await run('changePrivacy("memory",true)');
 assert.equal(consent.memory_enabled,true);assert.equal(consent.history_enabled,false);
 await run('changePrivacy("history",true)');
 await run('changePrivacy("history",false)');
 assert.equal(consent.memory_enabled,true);assert.equal(run('S.current.messages.length'),0);
 assert.equal(run('temporaryChats.size'),0);
 // A bad password is not an expired-session event; the dialog remains available.
 wrongPassword=true;run('accountDialog()');
 await assert.rejects(run('api("/accounts/login","POST",{alias:"synthetic",password:"wrong"})'));
 assert.equal(run('S.signedIn'),true);wrongPassword=false;
 // A late reply from a signed-out account must not appear in the next view.
 holdReply={};const pending=run('send("Pending fictional message")');
 for(let i=0;i<5&&!holdReply.resolve;i++)await tick();
 assert.ok(holdReply.resolve);await run('signOut()');holdReply.resolve();await pending;holdReply=null;
 assert.equal(run('S.signedIn'),false);assert.equal(run('S.current.messages.length'),0);
 assert.equal(run('temporaryChats.size'),0);assert.equal(run('S.draft'),'');
 await run('loadAccount()');expired=true;
 await assert.rejects(run('api("/accounts/me")'));
 assert.equal(run('S.signedIn'),false);assert.equal(run('S.userId'),null);
 assert.ok(!fs.readFileSync(path.join(root,'connection.js'),'utf8').match(/localStorage|sessionStorage|sample\(/));
 console.log(JSON.stringify({passed:true,contracts:['shipped_scripts_execute','real_api_requests','approved_only_display',
  'safe_plaintext_and_sources','temporary_chat_switch','encrypted_profile_api','independent_consents',
  'failed_settings_not_applied','late_reply_after_logout_discarded','session_expiry_clears_private_view'],browser_rendering_test:false}));
})().catch(error=>{console.error(error);process.exitCode=1;});
