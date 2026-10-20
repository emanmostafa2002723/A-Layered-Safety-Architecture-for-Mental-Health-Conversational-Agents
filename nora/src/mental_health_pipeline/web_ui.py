"""Dependency-free local chat page served by the FastAPI application."""

LOCAL_CHAT_HTML = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'self'; connect-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; base-uri 'none'; form-action 'self'">
  <title>Mental Health Pipeline — Local Test</title>
  <style>
    :root { color-scheme: light; --ink:#18302d; --muted:#60706d; --paper:#f4f7f5; --card:#fff; --accent:#28796f; --soft:#deefea; --danger:#8b2f35; }
    * { box-sizing:border-box; }
    body { margin:0; font:16px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; color:var(--ink); background:linear-gradient(145deg,#edf7f3,#f8f4ec); }
    main { width:min(900px,100%); min-height:100vh; margin:auto; padding:24px; display:grid; grid-template-rows:auto auto 1fr auto; gap:14px; }
    header, section, form { background:rgba(255,255,255,.94); border:1px solid #d9e5e1; border-radius:18px; box-shadow:0 10px 32px rgba(26,64,58,.08); }
    header { padding:20px 22px; }
    h1 { margin:0 0 5px; font-size:clamp(1.3rem,4vw,2rem); }
    p { margin:5px 0; }
    .notice { color:var(--danger); font-weight:650; }
    .controls { padding:12px 16px; display:flex; flex-wrap:wrap; gap:12px 20px; align-items:center; }
    label { display:flex; gap:8px; align-items:center; }
    select, textarea, button { font:inherit; }
    select, textarea { border:1px solid #afc4bf; border-radius:10px; background:#fff; color:var(--ink); }
    select { padding:7px 9px; }
    #status { margin-left:auto; color:var(--muted); font-size:.9rem; }
    #messages { padding:18px; overflow:auto; min-height:330px; max-height:58vh; display:flex; flex-direction:column; gap:12px; }
    .message { max-width:82%; padding:11px 14px; border-radius:15px; white-space:pre-wrap; overflow-wrap:anywhere; }
    .user { align-self:flex-end; background:var(--accent); color:white; border-bottom-right-radius:4px; }
    .assistant { align-self:flex-start; background:var(--soft); border-bottom-left-radius:4px; }
    .meta { display:block; margin-top:7px; color:var(--muted); font-size:.75rem; }
    form { padding:14px; display:grid; grid-template-columns:1fr auto; gap:10px; }
    textarea { width:100%; min-height:64px; max-height:180px; resize:vertical; padding:11px; }
    button { border:0; border-radius:11px; padding:10px 18px; background:var(--accent); color:white; font-weight:700; cursor:pointer; }
    button:disabled { opacity:.5; cursor:not-allowed; }
    #clear { background:#e8efed; color:var(--ink); padding:7px 11px; }
    @media(max-width:620px){ main{padding:10px}.message{max-width:94%}form{grid-template-columns:1fr}#status{width:100%;margin:0} }
  </style>
</head>
<body>
<main>
  <header>
    <h1>Mental Health Pipeline</h1>
    <p>Local integration test — crisis guard, boundary guard, response policy, RAG, generator, and output guard.</p>
    <p class="notice">Not an emergency service or a substitute for professional care. If there is immediate danger, contact local emergency services.</p>
  </header>
  <section class="controls" aria-label="Test settings">
    <label><input id="adult" type="checkbox"> I confirm this test is for an adult (18+)</label>
    <label>Confirmed location
      <select id="country">
        <option value="">Not provided</option>
        <option value="CA">Canada</option>
        <option value="US">United States</option>
      </select>
    </label>
    <button id="clear" type="button">New session</button>
    <span id="status">Checking service…</span>
  </section>
  <section id="messages" aria-live="polite">
    <div class="message assistant">The local test interface is ready. Confirm the adult scope, then type a message below.<span class="meta">System</span></div>
  </section>
  <form id="chat-form">
    <textarea id="message" maxlength="16384" placeholder="Type a message…" required aria-label="Message"></textarea>
    <button id="send" type="submit">Send</button>
  </form>
</main>
<script>
(() => {
  const messages = document.getElementById('messages');
  const form = document.getElementById('chat-form');
  const input = document.getElementById('message');
  const send = document.getElementById('send');
  const adult = document.getElementById('adult');
  const country = document.getElementById('country');
  const status = document.getElementById('status');
  let session = crypto.randomUUID();
  let turn = 1;

  function add(text, role, meta) {
    const item = document.createElement('div');
    item.className = `message ${role}`;
    item.textContent = text;
    if (meta) {
      const detail = document.createElement('span');
      detail.className = 'meta';
      detail.textContent = meta;
      item.appendChild(detail);
    }
    messages.appendChild(item);
    messages.scrollTop = messages.scrollHeight;
    return item;
  }

  async function health() {
    try {
      const response = await fetch('/health/ready', {cache:'no-store'});
      const data = await response.json();
      status.textContent = response.ok ? 'Service ready' : 'Fallback-only / components missing';
      status.title = (data.issues || []).join(', ');
    } catch (_) {
      status.textContent = 'Service unavailable';
    }
  }

  form.addEventListener('submit', async event => {
    event.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    if (!adult.checked) {
      add('Please confirm the adult (18+) test scope before sending.', 'assistant', 'Interface validation');
      return;
    }
    add(text, 'user', `Turn ${turn}`);
    input.value = '';
    send.disabled = true;
    const waiting = add('Processing the complete message…', 'assistant', 'No partial response is streamed');
    try {
      const response = await fetch('/v1/chat', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({
          message:text,
          session_id:session,
          turn_number:turn,
          age_eligibility_18_plus:true,
          preferred_language:'en',
          current_location_country:country.value || null,
          location_confirmed:Boolean(country.value)
        })
      });
      const data = await response.json();
      waiting.remove();
      if (!response.ok) throw new Error(data.detail || 'request_failed');
      add(data.response, 'assistant', `${data.route}${data.used_fallback ? ' · safe fallback' : ' · approved generation'}`);
      turn += 1;
    } catch (error) {
      waiting.remove();
      add('The local service could not complete that request. Check the server terminal and readiness status.', 'assistant', 'Request failed');
    } finally {
      send.disabled = false;
      input.focus();
    }
  });

  document.getElementById('clear').addEventListener('click', () => {
    session = crypto.randomUUID(); turn = 1;
    messages.replaceChildren();
    add('New local test session started.', 'assistant', 'System');
  });
  input.addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); form.requestSubmit(); }
  });
  health();
})();
</script>
</body>
</html>'''


__all__ = ["LOCAL_CHAT_HTML"]
