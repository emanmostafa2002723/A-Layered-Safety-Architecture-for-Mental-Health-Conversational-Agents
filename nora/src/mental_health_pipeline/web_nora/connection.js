'use strict';

// Same-origin account API. Tokens remain in HttpOnly cookies; no browser storage.
const temporaryChats=new Map();
let temporaryCharacters=0;
let chatSelection=0;
const TEMP_TTL=3600000,TEMP_CHARACTERS=16000,TEMP_PAIRS=12;
const errorMessages={
 sign_in_required:'Please sign in again to continue.',
 invalid_credentials:'The alias or password does not match. Please try again.',
 registration_unavailable:'That alias is unavailable. Please choose another.',
 use_a_3_to_32_character_alias:'Use 3–32 lowercase letters, numbers, hyphens or underscores for your alias.',
 password_must_be_12_to_128_characters:'Please use a password of 12–128 characters.',
 adult_confirmation_required:'Please confirm that you are 18 or older.',
 too_many_auth_attempts:'There have been several sign-in attempts. Please wait a minute before trying again.',
 stored_content_unavailable:'Some saved information could not be opened. Please try again later.',
 stored_content_binding_failed:'Some saved information could not be opened. Please try again later.',
 pipeline_no_checked_response:'I couldn’t prepare a checked response this time. Your message is still visible.',
 chat_not_found:'This conversation is no longer available. You can start a new one.',
 chat_limit_reached:'Your account has reached its conversation limit. Delete an old chat to make room.',
 invalid_request:'Please check the information you entered and try again.',
 invalid_request_origin:'Please reopen NORA at the address printed in its terminal.',
 service_unavailable:'The service is unavailable. Your message has not been automatically resent.',
 connection_interrupted:'The connection was interrupted. Your message has not been automatically resent.'
};

class WebError extends Error{
 constructor(code,status=0){super(errorMessages[code]||'The request could not be completed. Please try again.');this.code=code;this.status=status;}
}
class StaleSession extends Error {}

async function api(path,method='GET',body,options={}){
 const epoch=S.accountEpoch;
 const config={method,credentials:'same-origin',cache:'no-store',headers:{'Accept':'application/json'}};
 if(method!=='GET'){
  config.headers['Content-Type']='application/json';config.headers['X-Memory-Client']='1';
  config.body=JSON.stringify(body??{});
 }
 let response;
 try{response=await fetch(path,config);}catch{throw new WebError('connection_interrupted');}
 let data;
 try{data=await response.json();}catch{throw new WebError('service_unavailable',response.status);}
 if(epoch!==S.accountEpoch)throw new StaleSession();
 if(!response.ok){
  const code=typeof data.detail==='string'?data.detail:'invalid_request';
  if(response.status===401&&code==='sign_in_required'&&S.signedIn&&!options.initial){
   resetAccount();render();toast('Your session has ended. Please sign in again.');
  }
  throw new WebError(code,response.status);
 }
 return data;
}

function uiValues(patch={}){
 return {display_name:S.name,avatar:{...S.avatar},theme:S.theme,text:S.text,motion:S.motion,
  quiet:S.quiet,country:S.country||null,location_confirmed:!!S.country&&S.locationConfirmed,
  bookmarks:[...S.bookmarks],...patch};
}
function applyUI(values){
 S.name=values.display_name||'';S.avatar={...values.avatar};
 for(const key of ['theme','text','motion','quiet','bookmarks'])S[key]=values[key];
 S.country=values.country||'';S.locationConfirmed=values.location_confirmed;
}
function notesFromAPI(rows){
 return rows.map(n=>({id:n.id,text:'You '+n.user_report+'.',created:Date.parse(n.reported_on_utc+'T00:00:00Z'),expires:n.expires_at*1000}));
}
function chatsFromAPI(rows){
 return rows.map(c=>({id:c.id,created:c.created*1000,expires:c.expires*1000,turn:c.turn,
  title:temporaryChats.get(c.id)?.title||''}));
}
async function loadAccount({initial=false}={}){
 const epoch=S.accountEpoch;
 const profile=await api('/accounts/me','GET',undefined,{initial});
 const [prefs,chats,notes]=await Promise.all([
  api('/accounts/interface-preferences'),api('/accounts/chats'),api('/accounts/memories')
 ]);
 if(epoch!==S.accountEpoch)return;
 if(S.userId&&S.userId!==profile.id)resetAccount();
 S.signedIn=true;S.userId=profile.id;S.alias=profile.alias;
 S.memory=profile.memory_enabled;S.history=profile.history_enabled;
 applyUI(prefs);S.chats=chatsFromAPI(chats);S.notes=notesFromAPI(notes);S.ready=true;
}
async function refreshSaved(){
 if(!S.signedIn)return;
 const epoch=S.accountEpoch;
 const [chats,notes]=await Promise.all([api('/accounts/chats'),api('/accounts/memories')]);
 if(epoch===S.accountEpoch){S.chats=chatsFromAPI(chats);S.notes=notesFromAPI(notes);}
}
function requireAccount(){
 if(!S.ready){toast('Checking your session. Please wait a moment.');return false;}
 if(!S.signedIn){accountDialog();return false;}return true;
}
async function saveUI(patch){
 if(!requireAccount())return false;
 if(S.uiBusy){toast('Your previous choice is still being saved.');return false;}
 const epoch=S.accountEpoch;
 S.uiBusy=true;
 try{
  const values=await api('/accounts/interface-preferences','PUT',uiValues(patch));
  if(epoch!==S.accountEpoch)return false;
  applyUI(values);return true;
 }finally{if(epoch===S.accountEpoch)S.uiBusy=false;}
}
function pruneTemporary(){
 const now=Date.now();
 for(const [id,c] of temporaryChats)if(now-c.updated>TEMP_TTL)temporaryChats.delete(id);
 temporaryCharacters=[...temporaryChats.values()].reduce((n,c)=>n+c.size,0);
 while(temporaryCharacters>TEMP_CHARACTERS||temporaryChats.size>100){
  const id=temporaryChats.keys().next().value;
  if(id===undefined)break;
  temporaryCharacters-=temporaryChats.get(id).size;temporaryChats.delete(id);
 }
}
function rememberDisplayed(chat){
 if(!chat.id)return;
 const messages=chat.messages.slice(-TEMP_PAIRS*2).map(m=>({...m}));
 const size=JSON.stringify(messages).length;
 temporaryChats.delete(chat.id);
 if(size<=TEMP_CHARACTERS)temporaryChats.set(chat.id,{messages,size,updated:Date.now(),
  title:messages.find(m=>m.role==='user')?.text.slice(0,60)||''});
 pruneTemporary();
}
function displayedResponse(result){
 const released=result.text_released===true&&typeof result.response==='string'&&!!result.response.trim();
 if(!released)return {role:'assistant',status:true,text:'I couldn’t prepare a checked response this time. You can rephrase what you would like help with.',released:false};
 return {role:'assistant',text:result.response,released:true,citations:Array.isArray(result.citations)?result.citations:[],
  resources:result.resources||null};
}
function historyMessages(rows){
 return rows.flatMap(row=>[{role:'user',text:row.user},
  displayedResponse({...row.response,text_released:!!row.response.response?.trim()})]);
}
async function openChat(id){
 if(!requireAccount()||S.busy)return;
 pruneTemporary();
 const meta=S.chats.find(c=>c.id===id);if(!meta)return;
 const epoch=S.accountEpoch;
 const selection=++chatSelection;
 const rows=await api('/accounts/chats/'+encodeURIComponent(id));
 if(epoch!==S.accountEpoch||selection!==chatSelection||S.busy)return;
 S.current={...meta,messages:S.history?historyMessages(rows):(temporaryChats.get(id)?.messages.map(m=>({...m}))||[])};
 S.draft='';S.savedOpen=false;go('chat');
}
async function freshChat(){
 if(!requireAccount())return;
 if(S.busy){toast('Please wait for this reply before starting another conversation.');return;}
 const epoch=S.accountEpoch;
 const selection=++chatSelection;
 const chat=await api('/accounts/chats','POST',{});
 if(epoch!==S.accountEpoch||selection!==chatSelection||S.busy)return;
 S.current={id:chat.id,created:chat.created_at*1000,expires:chat.expires_at*1000,turn:0,messages:[]};
 S.chats.unshift({...S.current});S.draft='';S.savedOpen=false;go('chat');
 $('#chat-message')?.focus({preventScroll:true});
}
async function send(text){
 if(S.busy||!text.trim()||!requireAccount())return;
 const value=text.trim();
 if(new TextEncoder().encode(value).length>16384){toast('Please send a shorter message.');return;}
 const epoch=S.accountEpoch;S.busy=true;S.draft='';S.lastMessage=value;
 ++chatSelection;
 try{
  if(!S.current.id){
   const chat=await api('/accounts/chats','POST',{});
   if(epoch!==S.accountEpoch)return;
   S.current={id:chat.id,created:chat.created_at*1000,expires:chat.expires_at*1000,turn:0,messages:[]};
  }
  const chat=S.current;
  chat.messages.push({role:'user',text:value});
  S.waiting='Preparing and checking your response…';S.page='chat';render();
  slowTimer=setTimeout(()=>{if(epoch===S.accountEpoch&&S.busy){S.waiting='This is taking a little longer. Your response is still being prepared and checked.';render();}},12000);
  const result=await api('/accounts/chats/'+encodeURIComponent(chat.id)+'/messages','POST',{
   message:value,country:S.country||null,location_confirmed:!!S.country&&S.locationConfirmed
  });
  if(epoch!==S.accountEpoch)return;
  chat.messages.push(displayedResponse(result));chat.turn=(chat.turn||0)+1;
  rememberDisplayed(chat);
  try{await refreshSaved();}catch(error){if(epoch===S.accountEpoch&&S.signedIn)toast('Your reply arrived, but saved information could not be refreshed.');}
 }catch(error){
  if(epoch===S.accountEpoch&&S.signedIn){
   S.current.messages.push({role:'assistant',status:true,released:false,text:error.message});
   rememberDisplayed(S.current);
  }
 }finally{
  clearTimeout(slowTimer);
  if(epoch===S.accountEpoch){S.busy=false;S.waiting='';render();$('#chat-message')?.focus({preventScroll:true});}
 }
}
function resetAccount(){
 const epoch=S.accountEpoch+1;clearTimeout(slowTimer);stopSessionTimer();session=null;
 ++chatSelection;
 temporaryChats.clear();temporaryCharacters=0;
 Object.assign(S,{page:'home',name:'',alias:'',signedIn:false,ready:true,userId:null,accountEpoch:epoch,
  avatar:{character:2,colour:0,accessory:'none'},theme:'day',text:'default',motion:'auto',quiet:false,
  memory:false,history:false,notes:[],chats:[],current:{id:'',messages:[],created:Date.now()},draft:'',mood:'',
  bookmarks:[],country:'',locationConfirmed:false,savedOpen:false,showTitles:false,busy:false,uiBusy:false,waiting:'',lastMessage:'',
  resourceQuery:'',topic:'All',format:'All',onlySaved:false,exerciseFilter:'All',preference:'Listen'});
 if($('#modal').open)closeModal();
}
function applyLiveView(){
 const pw=$('#account-password');if(pw)pw.value='';
 if(!S.signedIn&&S.page==='space'){
  $$('[data-action="toggle-memory"],[data-action="toggle-history"]').forEach(b=>b.disabled=true);
 }
 if(S.busy){
  $$('[data-action="new-chat"],[data-action="toggle-memory"],[data-action="toggle-history"],[data-chat],[data-delete-chat],[data-forget],#profile-form button').forEach(b=>b.disabled=true);
 }
 if($('#home-message'))$('.home-composer .send').disabled=S.busy||!S.ready;
}
function accountDialog(mode='login'){
 if(S.accountBusy){toast('An account request is still being completed.');return;}
 if(S.signedIn){go('space');return;}
 modal(mode==='login'?'A familiar space.':'A space of your own.',`<div class="modal-tabs"><button data-auth="login" class="${mode==='login'?'active':''}">Sign in</button><button data-auth="register" class="${mode==='register'?'active':''}">Create account</button></div><p>${mode==='login'?'Sign in to return to your conversations and choices.':'An alias and a password are enough to begin. This service is for adults aged 18 and older.'}</p><form id="account-form" data-mode="${mode}"><div class="field"><label for="account-alias">Alias</label><input class="input" id="account-alias" maxlength="32" minlength="3" required autocomplete="username" placeholder="Your alias"></div><div class="field"><label for="account-password">Password</label><input class="input" id="account-password" type="password" required minlength="${mode==='register'?12:1}" maxlength="128" autocomplete="${mode==='register'?'new-password':'current-password'}" placeholder="${mode==='register'?'At least 12 characters':'Your password'}"></div>${mode==='register'?'<label class="checkbox-line"><input id="adult-confirmed" type="checkbox" required> I confirm I am 18 or older.</label><p class="hint">Memory and saved history start off. You can choose them in My space.</p>':''}<p class="error-text" id="account-error" role="status"></p><button class="button primary full-width" type="submit">${mode==='login'?'Sign in':'Create account'}</button></form>${mode==='login'?'<button class="text-button" data-action="recovery" style="margin-top:13px">Forgot password?</button>':''}`);
}
function memoryInfo(){
 if(!requireAccount())return;
 modal('What stays with you.',`<p>Personal memory can keep a few specific, dated notes about sleep, stress, improvement, and a preference for listening.</p><p>Notes expire 30 days after an explicit update. You can view and forget them anytime. Saved conversations are a separate choice.</p><div class="modal-footer"><button class="button quiet" data-action="close">Back to my space</button><button class="button primary" data-action="${S.memory?'open-space':'enable-memory'}">${S.memory?'View my notes':'Turn memory on'}</button></div>`);
}
function privacyToggle(which){
 if(!requireAccount()||S.busy)return;
 if(S[which])confirm(which==='memory'?'Turn off personal memory?':'Turn off saved history?',which==='memory'?'This deletes your saved memory notes. It does not delete the same words from separately saved conversations.':'This deletes saved transcripts and clears the current chat context. Personal memory notes are a separate choice.',`disable-${which}`,'Turn off and delete');
 else if(which==='memory')memoryInfo();
 else modal('Keep a conversation to return to.',`<p>Saved history keeps conversations for up to 30 days. Memory is a separate choice. You can turn saved history off and delete it anytime.</p><div class="modal-footer"><button class="button quiet" data-action="close">Keep history off</button><button class="button primary" data-action="enable-history">Turn history on</button></div>`);
}
async function changePrivacy(which,enabled){
 if(!requireAccount()||S.busy||S.uiBusy)return;
 const epoch=S.accountEpoch;
 S.uiBusy=true;
 try{
 const profile=await api('/accounts/preferences','PUT',{
  memory_enabled:which==='memory'?enabled:S.memory,history_enabled:which==='history'?enabled:S.history
 });
 if(epoch!==S.accountEpoch)return;
 S.memory=profile.memory_enabled;S.history=profile.history_enabled;
 if(!S.memory)S.notes=[];
 if(which==='history'&&!enabled){temporaryChats.clear();S.current.messages=[];S.draft='';}
 await refreshSaved();closeModal();render();
 toast(which==='memory'?(enabled?'Personal memory is on. You can view and forget notes.':'Memory is off. Saved notes have been deleted.'):(enabled?'Saved history is on.':'Saved transcripts and active chat context have been cleared.'));
 }finally{if(epoch===S.accountEpoch)S.uiBusy=false;}
}
async function helpDialog(){
 const query=new URLSearchParams();if(S.country)query.set('country',S.country);query.set('location_confirmed',String(!!S.country&&S.locationConfirmed));
 const data=await api('/support-resources?'+query);
 const rows=(data.resources||[]).map(r=>{const link=safeLink(r.source_url);return `<div class="support-card"><h3>${esc(r.name)}</h3><p>${esc(r.action)}</p>${link?`<a class="text-button" href="${esc(link)}" target="_blank" rel="noopener noreferrer">Open official resource ↗</a>`:''}</div>`;}).join('');
 modal('You can reach someone.',`<p>${esc(data.immediate_danger_instruction)}</p>${rows}<p class="modal-small" style="margin-top:18px">${data.country_code?'Resources are based on your confirmed country.':'No current country has been assumed. Choose your country to find relevant support.'} This service does not contact anyone on your behalf.</p><div class="modal-footer"><button class="button quiet" data-action="open-space">Country settings</button><button class="button primary" data-action="close">Keep talking here</button></div>`);
}
function welcome(step=0){
 S.onboardingStep=step;
 const body=[`<div class="onboarding-art"><img src="${assetUrl('nora-logo.png')}" alt="NORA logo"></div><p>A place to talk things through, try a gentle activity, or learn something helpful. Choose where to begin.</p><div class="feature-line"><span class="tile-icon">${icon('chat')}</span><div><strong>A conversation, at your pace</strong>An AI companion for adults, in English.</div></div><div class="feature-line"><span class="tile-icon blue">${icon('wind')}</span><div><strong>A moment to pause</strong>Optional activities, with room to skip or stop.</div></div>`,`<p>Memory and saved history are separate choices. Both start off. Each can be changed in My space.</p><p>Memory keeps a few relevant notes. Saved history keeps the conversation itself. Both have a maximum 30-day retention period.</p>`,`<p>Choose your name and a little character whenever you like. There is no setup you have to finish before talking.</p><div class="profile-preview">${avatar(S.avatar,78)}<div><strong>${names[S.avatar.character]}</strong><br><button class="text-button" data-action="avatar">Choose a character</button></div></div>`][step];
 modal(['Meet your little clearing.','Your choices come first.','Make it feel like you.'][step],`<div class="steps" aria-label="Step ${step+1} of 3">${[0,1,2].map(i=>`<span class="${i<=step?'active':''}"></span>`).join('')}</div>${body}<div class="modal-footer"><button class="button quiet" data-action="close">Explore at my pace</button><button class="button primary" data-action="${step<2?'welcome-next':S.signedIn?'open-space':'signin'}">${step<2?'Continue':S.signedIn?'My space':'Sign in to begin'}</button></div><p class="modal-small" style="margin-top:17px">NORA is an AI companion, not a human therapist or emergency service.</p>`);
}
async function exportData(){
 if(!requireAccount())return;
 const data=await api('/accounts/export');
 const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
 const link=document.createElement('a');link.href=url;link.download='nora-my-data.json';link.click();
 setTimeout(()=>URL.revokeObjectURL(url),1000);toast('Your data has been downloaded. Keep the file somewhere private.');
}
async function signOut(){
 if(!S.signedIn)return;
 S.accountBusy=true;resetAccount();render();
 try{await api('/accounts/logout','POST',{});toast('Signed out.');}
 catch(error){toast('The page is cleared. Sign-out could not be confirmed; close the browser if you are on a shared device.');}
 finally{S.accountBusy=false;}
}
async function runAction(task){try{await task();}catch(error){if(!(error instanceof StaleSession))toast(error instanceof WebError?error.message:'The action could not be completed. Please try again.');}}

document.addEventListener('click',event=>{const b=event.target.closest('button,a');if(!b)return;
 if(b.dataset.page){event.preventDefault();go(b.dataset.page);return;}
 if(b.dataset.start){runAction(()=>send(b.dataset.start));return;}
 if(b.dataset.mood){S.mood=b.dataset.mood;render();toast('Noted for this moment. No mood history is saved.');return;}
 if(b.hasAttribute('data-topic')){S.topic=b.dataset.topic;render();return;}
 if(b.dataset.exerciseFilter){S.exerciseFilter=b.dataset.exerciseFilter;render();return;}
 if(b.dataset.resource){resourceDialog(b.dataset.resource);return;}
 if(b.dataset.bookmark){runAction(async()=>{if(!requireAccount())return;const id=b.dataset.bookmark;const bookmarks=S.bookmarks.includes(id)?S.bookmarks.filter(x=>x!==id):[...S.bookmarks,id];if(await saveUI({bookmarks})){render();if(b.dataset.modalBookmark)resourceDialog(id);toast(bookmarks.includes(id)?'Saved to your account.':'Removed from saved resources.');}});return;}
 if(b.dataset.exercise){exerciseDetails(b.dataset.exercise);return;}
 if(b.dataset.beginExercise){beginExercise(b.dataset.beginExercise);return;}
 if(b.dataset.pref){runAction(async()=>{if(S.signedIn)await saveUI({[b.dataset.pref]:b.dataset.value});else S[b.dataset.pref]=b.dataset.value;render();});return;}
 if(b.dataset.auth){accountDialog(b.dataset.auth);return;}
 if(b.hasAttribute('data-avatar-character')){avatarDraft.character=Number(b.dataset.avatarCharacter);drawAvatarDialog();return;}
 if(b.hasAttribute('data-avatar-colour')){avatarDraft.colour=Number(b.dataset.avatarColour);drawAvatarDialog();return;}
 if(b.dataset.avatarAccessory){avatarDraft.accessory=b.dataset.avatarAccessory;drawAvatarDialog();return;}
 if(b.dataset.chat){runAction(()=>openChat(b.dataset.chat));return;}
 if(b.dataset.deleteChat){if(S.busy)return;confirm('Delete this conversation?','This permanently deletes its transcript and chat context. Personal memory notes are separate.',`delete-chat:${b.dataset.deleteChat}`,'Delete conversation');return;}
 if(b.dataset.forget){confirm('Forget this note?','The saved note will be deleted. The same words may still appear in a separately saved conversation.',`forget:${b.dataset.forget}`,'Forget note');return;}
 const action=b.dataset.action;if(!action)return;event.preventDefault();
 runAction(async()=>{
  if(action.startsWith('delete-chat:')){await api('/accounts/chats/'+encodeURIComponent(action.slice(12)),'DELETE',{});if(S.current.id===action.slice(12))S.current={id:'',messages:[],created:Date.now()};temporaryChats.delete(action.slice(12));await refreshSaved();closeModal();render();return;}
  if(action.startsWith('forget:')){await api('/accounts/memories/'+encodeURIComponent(action.slice(7)),'DELETE',{});await refreshSaved();closeModal();render();return;}
  switch(action){
   case 'new-chat':case 'start-talk':await freshChat();break;
   case 'start-listen':await send('I just want someone to listen.');break;
   case 'open-exercises':go('exercises');break;case 'open-resources':go('resources');break;
   case 'open-space':closeModal();go('space');break;
   case 'theme':{const theme=S.theme==='day'?'evening':'day';if(S.signedIn)await saveUI({theme});else S.theme=theme;render();break;}
   case 'toggle-quiet':if(S.signedIn)await saveUI({quiet:!S.quiet});else S.quiet=!S.quiet;render();break;
   case 'saved-panel':S.savedOpen=!S.savedOpen;render();break;
   case 'memory-info':memoryInfo();break;case 'toggle-memory':privacyToggle('memory');break;case 'toggle-history':privacyToggle('history');break;
   case 'enable-memory':await changePrivacy('memory',true);break;case 'disable-memory':await changePrivacy('memory',false);break;
   case 'enable-history':await changePrivacy('history',true);break;case 'disable-history':await changePrivacy('history',false);break;
   case 'close':closeModal();break;case 'help':await helpDialog();break;
   case 'avatar':avatarDialog();break;
   case 'apply-avatar':{const selected={...avatarDraft};if(S.signedIn){if(!await saveUI({avatar:selected}))return;}else S.avatar=selected;closeModal();render();toast(S.signedIn?'Your avatar is saved.':'Your avatar is set for this visit. Sign in to keep it.');break;}
   case 'account':case 'signin':accountDialog();break;
   case 'recovery':modal('Account recovery.',`<p>Password recovery is not available in this private test yet. No recovery email can be sent from this screen.</p><div class="modal-footer"><button class="button primary" data-auth="login">Back to sign in</button></div>`);break;
   case 'welcome':welcome();break;case 'welcome-next':welcome(Math.min(2,S.onboardingStep+1));break;
   case 'saved-resources':S.onlySaved=!S.onlySaved;render();break;case 'clear-filters':S.topic='All';S.format='All';S.resourceQuery='';S.onlySaved=false;render();break;
   case 'pause-session':pauseSession();break;case 'stop-session':finishSession();break;case 'leave-exercise':session=null;stopSessionTimer();go('exercises');break;
   case 'next-step':if(session){const length=session.id==='grounding'?grounding.length:unwind.length;if(session.step<length-1){session.step++;render();}else finishSession();}break;
   case 'forget-all':confirm('Forget all notes?','This permanently deletes saved memory notes. Saved conversations are separate.','confirm-forget-all','Forget all notes');break;
   case 'confirm-forget-all':await api('/accounts/memories','DELETE',{});await refreshSaved();closeModal();render();break;
   case 'export':await exportData();break;case 'signout':await signOut();break;
   case 'delete-account':if(requireAccount())confirm('Delete your account and data?','This permanently deletes your account, sessions, conversations, memory notes and saved profile choices. This cannot be undone.','confirm-delete-account','Delete account and data');break;
   case 'confirm-delete-account':await api('/accounts/me','DELETE',{});resetAccount();render();toast('Your account and data have been deleted.');break;
  }
 });
});

document.addEventListener('submit',event=>{
 const form=event.target;event.preventDefault();
 if(form.id==='home-form'||form.id==='chat-form'){runAction(()=>send($(form.id==='home-form'?'#home-message':'#chat-message').value));return;}
 if(form.id==='profile-form'){runAction(async()=>{const name=$('#display-name').value.trim(),country=$('#country').value;const confirmed=!!country&&$('#country-confirmed').checked;if(await saveUI({display_name:name,country:country||null,location_confirmed:confirmed})){render();toast('Your profile choices are saved.');}});return;}
 if(form.id==='account-form'){
  if(S.accountBusy)return;S.accountBusy=true;
  const alias=$('#account-alias').value.trim(),password=$('#account-password').value,mode=form.dataset.mode;
  const adult=$('#adult-confirmed')?.checked===true;
  const submit=form.querySelector('button[type="submit"]');if(submit)submit.disabled=true;
  $('#account-error').textContent='';
  (async()=>{
   try{
    if(mode==='register')await api('/accounts/register','POST',{alias,password,adult_confirmed:adult});
    await api('/accounts/login','POST',{alias,password});
    $('#account-password').value='';S.accountEpoch++;await loadAccount();closeModal();S.page='home';render();toast('You are signed in. Your memory and history choices are in My space.');
   }catch(error){if($('#account-password'))$('#account-password').value='';if($('#account-error'))$('#account-error').textContent=error.message;}
   finally{S.accountBusy=false;if(submit)submit.disabled=false;}
  })();
 }
});
document.addEventListener('input',event=>{
 if(event.target.id==='home-message'||event.target.id==='chat-message')S.draft=event.target.value;
 if(event.target.id==='reflection'&&session)session.journal=event.target.value;
 if(event.target.id==='resource-search'){S.resourceQuery=event.target.value;$('#resource-results').innerHTML=resourceResults();hydrate($('#resource-results'));}
 if(event.target.id==='country')$('#country-confirmed').checked=false;
});
document.addEventListener('change',event=>{
 if(event.target.id==='resource-format'){S.format=event.target.value;$('#resource-results').innerHTML=resourceResults();hydrate($('#resource-results'));}
 if(event.target.dataset.setting==='titles'){S.showTitles=event.target.checked;render();}
});
document.addEventListener('keydown',event=>{if(['home-message','chat-message'].includes(event.target.id)&&event.key==='Enter'&&!event.shiftKey&&!event.isComposing){event.preventDefault();runAction(()=>send(event.target.value));}});
document.addEventListener('focusin',event=>{if(['home-message','chat-message'].includes(event.target.id))document.body.classList.add('keyboard-open');});
document.addEventListener('focusout',event=>{if(['home-message','chat-message'].includes(event.target.id))document.body.classList.remove('keyboard-open');});
$('#modal').addEventListener('close',()=>{if(!$('#modal').open){$('#modal-content').replaceChildren();avatarDraft=null;}});
$('#modal').addEventListener('click',event=>{if(event.target===$('#modal')){const r=$('#modal').getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)closeModal();}});
window.addEventListener('pagehide',()=>{clearTimeout(slowTimer);stopSessionTimer();temporaryChats.clear();});

async function startNora(){
 hydrate();render();
 try{await loadAccount({initial:true});}catch(error){if(error.status!==401)toast(error.message);}
 finally{S.ready=true;render();}
}
startNora();
