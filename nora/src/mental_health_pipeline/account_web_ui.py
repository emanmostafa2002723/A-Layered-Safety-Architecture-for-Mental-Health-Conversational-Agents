"""Small first-party account test UI. No CDN, analytics, raw HTML chat or token storage."""

ACCOUNT_HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Private account memory test</title>
<style>
body{font:17px system-ui,sans-serif;max-width:1050px;margin:30px auto;padding:0 20px;color:#182b35;background:#f3f6f7}
section{background:white;border:1px solid #ccd9df;border-radius:12px;padding:20px;margin:16px 0}
button,input,select,textarea{font:inherit;padding:9px;margin:5px;border:1px solid #afbec6;border-radius:6px}
button{cursor:pointer;background:#e3f0ec}button:disabled{opacity:.5;cursor:wait}textarea{width:90%;min-height:80px}
#messages{max-height:480px;overflow:auto;white-space:pre-wrap}.turn{padding:12px;border-bottom:1px solid #dde5e8}
.muted{color:#516470;font-size:15px}.danger{background:#ffeded}#notice{white-space:pre-wrap}label{display:block;margin:10px 0}
</style>
<h1>Account memory — private test</h1>
<p>Development prototype, not a clinically approved service. Keep this on localhost.
The existing Gradio diagnostic interface is unchanged.</p>
<p class="muted">Use fictional test data only. This development folder may sync through OneDrive;
real mental-health records and encryption keys must not be synced together.</p>
<section id="auth"><h2>Sign in or create an adult account</h2>
<p class="muted">Use a fictional alias for tests, not your real name. Password: 12–128 characters.
No email verification or password recovery is implemented.</p>
<input id="alias" autocomplete="username" aria-label="Alias" placeholder="test-alias">
<input id="password" type="password" autocomplete="current-password" aria-label="Password" placeholder="Password">
<label><input id="adult" type="checkbox">I confirm I am 18 or older (required to create an account).</label>
<button id="register">Create account</button><button id="login">Sign in</button></section>
<p id="notice" role="status" aria-live="polite"></p>
<main id="workspace" hidden>
<section><h2 id="identity">Your account</h2>
<label><input id="memory" type="checkbox">Remember explicit sleep/stress reports and listening preference for 30 days.</label>
<label><input id="history" type="checkbox">Save my chats for up to 30 days (encrypted content on this device).</label>
<p class="muted">Both start OFF. Switching memory off deletes saved notes. Switching history off deletes saved transcripts.
Without saved history, recent chat context is temporary. Notes can be imperfect and should be corrected by telling the assistant what has changed.
No diagnosis, medication instruction or crisis label is automatically saved as a personal note.</p>
<button id="save">Save preferences</button><button id="logout">Sign out</button>
<button id="export">Download my data</button><button class="danger" id="delete-account">Delete account and data</button></section>
<section><h2>Your chats</h2><select id="chats" aria-label="Choose a chat"></select>
<button id="new">New chat</button><button class="danger" id="delete-chat">Delete selected chat</button>
<div id="messages" aria-live="polite"></div>
<label>Current country code (optional): <input id="country" maxlength="2" size="3" placeholder="CA"></label>
<label><input id="location" type="checkbox">I confirm that is my current country.</label>
<textarea id="message" placeholder="Write a message" aria-label="Message"></textarea><br><button id="send">Send</button>
<p class="muted">Only fully checked replies are shown; a withheld response is marked without exposing its draft.
Switching chats does not carry crisis state into another chat.</p></section>
<section><h2>What is remembered</h2><div id="notes"></div>
<button id="refresh">Refresh notes</button><button class="danger" id="forget">Forget all notes</button>
<p class="muted">Forgetting notes does not delete saved chats. Delete a chat or turn saved history off to remove transcripts as well.</p></section>
</main>
<script>
'use strict';
const $=id=>document.getElementById(id); let selected='', busy=false, historyEnabled=false;
const temporaryChats=new Map();
function signedOut(){selected='';historyEnabled=false;temporaryChats.clear();
 $('messages').replaceChildren();$('notes').replaceChildren();$('chats').replaceChildren();
 $('message').value='';$('password').value='';$('identity').textContent='Your account';
 $('workspace').hidden=true;$('auth').hidden=false}
function temporaryRows(id){const entry=temporaryChats.get(id);if(!entry)return [];
 if(Date.now()-entry.at>=3600000){temporaryChats.delete(id);return []}return entry.rows}
function rememberDisplayed(id,row){const rows=[...temporaryRows(id),row];
 while(rows.length>12||JSON.stringify(rows).length>16000)rows.shift();
 temporaryChats.delete(id);temporaryChats.set(id,{at:Date.now(),rows});
 while(temporaryChats.size>100)temporaryChats.delete(temporaryChats.keys().next().value)}
async function api(path,method='GET',body){
 const opts={method,credentials:'same-origin',headers:{'X-Memory-Client':'1','Content-Type':'application/json'}};
 if(body!==undefined)opts.body=JSON.stringify(body);
 const r=await fetch('/accounts'+path,opts);const j=await r.json();
 if(r.status===401)signedOut();
 if(!r.ok)throw new Error(j.detail||'Request failed');return j;
}
function note(t){$('notice').textContent=t}
function show(role,text){const d=document.createElement('div');d.className='turn';d.textContent=role+': '+text;$('messages').append(d);d.scrollIntoView({block:'nearest'})}
function showResponse(result){show('Assistant',result.response||'[Response withheld by safety checks]');
 if(result.response&&(result.citations||[]).length){const lines=result.citations.map(c=>'['+c.source_id+'] '+c.title+' — '+c.source+(c.page?' (page '+c.page+')':'')+(c.url?' — '+c.url:''));show('Supplied sources',lines.join('\n'))}}
async function notes(){const rows=await api('/memories');$('notes').replaceChildren();
 if(!rows.length)$('notes').textContent='No saved notes.';
 for(const row of rows){const p=document.createElement('p');p.textContent=row.user_report+' — reported '+row.reported_on_utc+', expires '+new Date(row.expires_at*1000).toLocaleDateString()+' ';
 const b=document.createElement('button');b.textContent='Forget';b.onclick=()=>action(async()=>{await api('/memories/'+encodeURIComponent(row.id),'DELETE');await notes()});p.append(b);$('notes').append(p)}
}
async function loadChat(){selected=$('chats').value;$('messages').replaceChildren();if(!selected)return;
 const saved=await api('/chats/'+encodeURIComponent(selected));
 const rows=historyEnabled?saved:temporaryRows(selected);
 for(const row of rows){show('You',row.user);showResponse(row.response)}
}
async function chats(){const rows=await api('/chats');$('chats').replaceChildren();
 for(const row of rows){const o=document.createElement('option');o.value=row.id;o.textContent='Chat '+row.id.slice(0,8)+' — '+new Date(row.created*1000).toLocaleString();$('chats').append(o)}
 if(rows.some(r=>r.id===selected))$('chats').value=selected;await loadChat();
}
async function signedIn(){const me=await api('/me');$('auth').hidden=true;$('workspace').hidden=false;$('identity').textContent='Signed in: '+me.alias;
 $('memory').checked=me.memory_enabled;$('history').checked=me.history_enabled;historyEnabled=me.history_enabled;await chats();await notes()}
async function action(fn){if(busy)return;busy=true;document.querySelectorAll('button,input,select,textarea').forEach(e=>e.disabled=true);
 try{await fn()}catch(e){note(e.message)}finally{busy=false;document.querySelectorAll('button,input,select,textarea').forEach(e=>e.disabled=false)}}
$('register').onclick=()=>action(async()=>{await api('/register','POST',{alias:$('alias').value,password:$('password').value,adult_confirmed:$('adult').checked});note('Account created. Sign in next. Both storage preferences are OFF.')});
$('login').onclick=()=>action(async()=>{await api('/login','POST',{alias:$('alias').value,password:$('password').value});$('password').value='';await signedIn();note('Signed in. Choose storage preferences, then create a chat.')});
$('logout').onclick=()=>action(async()=>{await api('/logout','POST',{});location.reload()});
$('save').onclick=()=>action(async()=>{const me=await api('/preferences','PUT',{memory_enabled:$('memory').checked,history_enabled:$('history').checked});
 if(historyEnabled&&!me.history_enabled)temporaryChats.clear();historyEnabled=me.history_enabled;
 await chats();await notes();note('Preferences saved. Notes and saved chats expire after 30 days.')});
$('new').onclick=()=>action(async()=>{const row=await api('/chats','POST',{});selected=row.id;await chats();note('New chat created.')});
$('chats').onchange=()=>action(loadChat);
$('send').onclick=()=>action(async()=>{if(!selected)throw new Error('Create or choose a chat first.');const message=$('message').value.trim();if(!message)return;
 const country=$('country').value.trim().toUpperCase()||null;note('Running the full guarded pipeline. Please wait…');
 const r=await api('/chats/'+encodeURIComponent(selected)+'/messages','POST',{message,country,location_confirmed:$('location').checked});
 if(!historyEnabled)rememberDisplayed(selected,{user:message,response:r});
 show('You',message);showResponse(r);$('message').value='';
 note('Route: '+r.route+' | Dated memory notes supplied: '+r.memory_notes_supplied+' | Text released: '+r.text_released);await notes()});
$('refresh').onclick=()=>action(notes);
$('forget').onclick=()=>action(async()=>{if(confirm('Delete every saved personal note? Saved chats are separate.')){await api('/memories','DELETE');await notes()}});
$('delete-chat').onclick=()=>action(async()=>{if(selected&&confirm('Delete this chat and its saved transcript? Personal notes are separate.')){await api('/chats/'+encodeURIComponent(selected),'DELETE');temporaryChats.delete(selected);selected='';await chats()}});
$('delete-account').onclick=()=>action(async()=>{if(confirm('Permanently delete this account, sessions, chats and personal notes?')){await api('/me','DELETE');location.reload()}});
$('export').onclick=()=>action(async()=>{const data=await api('/export');const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
 const a=document.createElement('a');a.href=url;a.download='my-private-chat-data.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);note('Private export downloaded. Store it securely; it contains decrypted account data.')});
signedIn().catch(()=>{});
</script></html>'''
