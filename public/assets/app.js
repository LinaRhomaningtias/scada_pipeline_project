const $ = (id) => document.getElementById(id);
const pages = document.querySelectorAll('.page');
const navs = document.querySelectorAll('.nav-btn');

function showPage(name){
  pages.forEach(p => p.classList.toggle('active-page', p.id === name));
  navs.forEach(n => n.classList.toggle('active', n.dataset.page === name));
  const titles = {dashboard:'SCADA Pipeline Monitoring',dataset:'Dataset SCADA',prediction:'Pipeline Condition Detection',team:'Our Team'};
  $('page-title').textContent = titles[name];
  window.scrollTo({top:0,behavior:'smooth'});
}
navs.forEach(n => n.addEventListener('click', () => showPage(n.dataset.page)));

function pct(v){ return `${(v*100).toFixed(2)}%`; }

async function loadDashboard(){
  try{
    const m = await fetch('/api/metrics').then(r=>r.json());
    $('total-data').textContent = (m.normal_count + m.anomaly_count).toLocaleString('id-ID');
    $('feature-count').textContent = m.feature_count;
    $('normal-count').textContent = m.normal_count.toLocaleString('id-ID');
    $('anomaly-count').textContent = m.anomaly_count.toLocaleString('id-ID');
    $('normal-percent').textContent = pct(m.normal_count/(m.normal_count+m.anomaly_count));
    $('anomaly-percent').textContent = pct(m.anomaly_count/(m.normal_count+m.anomaly_count));
    $('accuracy').textContent = pct(m.accuracy); $('precision').textContent = pct(m.precision); $('recall').textContent = pct(m.recall); $('f1').textContent = pct(m.f1);
    $('donut-total').textContent = (m.normal_count+m.anomaly_count).toLocaleString('id-ID');
    $('legend-normal').textContent = m.normal_count; $('legend-anomaly').textContent = m.anomaly_count;
  }catch(e){console.error(e)}
  try{
    const rows = await fetch('/api/recent').then(r=>r.json());
    $('recent-body').innerHTML = rows.map(r=>`<tr><td>${r.timestamp}</td><td>${r.pressure}</td><td>${r.flow_rate}</td><td>${r.temperature}</td><td><span class="pill ${r.target.toLowerCase()}">${r.target}</span></td></tr>`).join('');
  }catch(e){console.error(e)}
}

async function predict(){
  const ids=['pressure','flow_rate','temperature','valve_status','pump_state','pump_speed','compressor_state','energy_consumption'];
  const payload={}; ids.forEach(id=>payload[id]=Number($(id).value));
  const btn=document.querySelector('.form-card .primary'); btn.disabled=true; btn.textContent='Processing...';
  try{
    const res=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    if(!res.ok) throw new Error('Prediction failed');
    const d=await res.json(); const anomaly=d.prediction===1; const card=$('result-card');
    card.classList.toggle('anomaly',anomaly); card.classList.toggle('normal',!anomaly);
    $('result-label').textContent=anomaly?'ANOMALY':'NORMAL';
    $('result-description').textContent=anomaly?'Pipeline terindikasi memiliki kondisi anomali berdasarkan parameter input.':'Pipeline diprediksi berada dalam kondisi normal.';
    $('normal-prob').textContent=pct(d.normal_probability); $('anomaly-prob').textContent=pct(d.anomaly_probability);
  }catch(e){alert('Gagal melakukan prediksi. Pastikan API/model sudah berjalan.');console.error(e)}
  finally{btn.disabled=false;btn.textContent='⌕ Detect Condition';}
}

async function loadTeam(){
  try{
    const data=await fetch('/api/team').then(r=>r.json());
    $('github-link').href=data.github_project;
    $('team-github').href=data.github_project;
    $('team-grid').innerHTML=data.members.map(x=>`<div class="team-card"><div class="avatar">${x.name.split(' ').map(v=>v[0]).slice(0,2).join('')}</div><h3>${x.name}</h3><p>${x.role}</p><div class="socials"><a href="${x.linkedin}" target="_blank" rel="noopener">LinkedIn ↗</a><a href="${x.github}" target="_blank" rel="noopener">GitHub ↗</a></div></div>`).join('');
  }catch(e){console.error(e)}
}

loadDashboard(); loadTeam();
