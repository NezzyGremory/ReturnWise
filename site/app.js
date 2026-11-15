(() => {
  'use strict';
  const $ = (selector) => document.querySelector(selector);
  const fileInput = $('#csv-file');
  const dropzone = $('.dropzone');
  const fileStatus = $('#file-status');
  const profile = $('#profile');
  const queueSelect = $('#capacity-select');
  const customInput = $('#capacity-custom');
  const search = $('#queue-search');
  const menuToggle = $('.menu-toggle');
  const nav = $('#primary-nav');
  const demoRows = [
    {order_id:'RW-2048', region:'North', category:'Home', probability:0.78},
    {order_id:'RW-1175', region:'West', category:'Apparel', probability:0.71},
    {order_id:'RW-3062', region:'Central', category:'Electronics', probability:0.67},
    {order_id:'RW-2419', region:'South', category:'Home', probability:0.61},
    {order_id:'RW-0851', region:'East', category:'Apparel', probability:0.54},
    {order_id:'RW-1933', region:'West', category:'Beauty', probability:0.48},
    {order_id:'RW-2790', region:'North', category:'Electronics', probability:0.36},
    {order_id:'RW-3341', region:'Central', category:'Home', probability:0.22}
  ];
  let rows = demoRows.map((row) => ({...row}));
  let headers = ['order_id','region','category','return_probability'];
  let probabilityKey = 'probability';
  let source = 'demo';
  let currentFile = null;
  let currentCapacity = 25;

  $('#year').textContent = new Date().getFullYear();
  menuToggle.addEventListener('click', () => {
    const open = menuToggle.getAttribute('aria-expanded') !== 'true';
    menuToggle.setAttribute('aria-expanded', String(open));
    nav.classList.toggle('open', open);
  });
  nav.addEventListener('click', (event) => {
    if (event.target.closest('a')) { nav.classList.remove('open'); menuToggle.setAttribute('aria-expanded','false'); }
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && nav.classList.contains('open')) {
      nav.classList.remove('open');
      menuToggle.setAttribute('aria-expanded', 'false');
      menuToggle.focus();
    }
  });
  fileInput.addEventListener('change', () => { if (fileInput.files[0]) loadFile(fileInput.files[0]); });
  dropzone.addEventListener('dragover', (event) => { event.preventDefault(); dropzone.classList.add('dragging'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragging'));
  dropzone.addEventListener('drop', (event) => { event.preventDefault(); dropzone.classList.remove('dragging'); const file = event.dataTransfer.files[0]; if (file) loadFile(file); });
  $('#clear-file').addEventListener('click', resetFile);
  queueSelect.addEventListener('change', updateCapacity);
  customInput.addEventListener('input', updateCapacity);
  search.addEventListener('input', renderQueue);
  $('#download-queue').addEventListener('click', downloadQueue);

  function updateCapacity() {
    const custom = queueSelect.value === 'custom';
    $('.custom-capacity[for="capacity-custom"]').hidden = !custom;
    customInput.hidden = !custom;
    currentCapacity = Math.max(1, Math.min(10000, Number(custom ? customInput.value : queueSelect.value) || 1));
    renderQueue();
  }

  function escapeHTML(value) {
    return String(value ?? '').replace(/[&<>"']/g, (char) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  }
  function parseCSV(text) {
    const result = [];
    let row = [], field = '', quoted = false;
    for (let i=0; i<text.length; i++) {
      const ch = text[i];
      if (quoted) {
        if (ch === '"' && text[i+1] === '"') { field += '"'; i++; }
        else if (ch === '"') quoted = false;
        else field += ch;
      } else if (ch === '"') quoted = true;
      else if (ch === ',') { row.push(field); field = ''; }
      else if (ch === '\n') { row.push(field.replace(/\r$/, '')); result.push(row); row=[]; field=''; }
      else field += ch;
    }
    if (field.length || row.length) { row.push(field.replace(/\r$/, '')); result.push(row); }
    return result.filter((line) => line.some((value) => value.trim() !== ''));
  }
  function normalizeProbability(header) {
    const key = header.trim().toLowerCase().replace(/[\s-]+/g,'_');
    if (['calibrated_probability','return_probability','probability_return','p_return','return_prob'].includes(key)) return key;
    return null;
  }
  function loadFile(file) {
    currentFile = file;
    fileStatus.className = 'message';
    rows = [];
    source = 'unavailable';
    profile.hidden = true;
    $('#queue-mode').textContent = 'INFERENCE UNAVAILABLE';
    $('#queue-mode').className = 'demo-pill unavailable-pill';
    $('#queue-disclaimer').textContent = 'No risk scores are generated here. Use the local Streamlit research app for experimental Python scoring.';
    renderQueue();
    if (!file.name.toLowerCase().endsWith('.csv') && file.type !== 'text/csv') return setError('Choose a CSV file to continue.');
    if (file.size > 25 * 1024 * 1024) return setError('This file is larger than 25 MB. Choose a smaller CSV.');
    fileStatus.textContent = 'Reading file in this browser…';
    const reader = new FileReader();
    reader.onerror = () => setError('The file could not be read. Try exporting it as UTF-8 CSV.');
    reader.onload = () => {
      try {
        const parsed = parseCSV(String(reader.result ?? '').replace(/^\uFEFF/,''));
        if (parsed.length < 2) return setError('This CSV needs a header row and at least one data row.');
        headers = parsed[0].map((value, index) => value.trim() || `column_${index + 1}`);
        const data = parsed.slice(1).map((line) => headers.map((_, index) => line[index] ?? '')).filter((line) => line.some((value) => value.trim() !== ''));
        if (data.length > 100000) return setError('This local inspector supports up to 100,000 data rows per file.');
        const normalized = headers.map(normalizeProbability);
        const probabilityIndex = normalized.findIndex(Boolean);
        const probabilityHeader = probabilityIndex >= 0 ? normalized[probabilityIndex] : null;
        const tableRows = data.map((line) => Object.fromEntries(headers.map((header, index) => [header, line[index]])));
        const hasTarget = headers.some((header) => ['returned','return_label','is_returned','target'].includes(header.trim().toLowerCase()));
        const hasValidScores = probabilityIndex >= 0 && data.every((line) => line[probabilityIndex].trim() !== '' && Number.isFinite(Number(line[probabilityIndex])) && Number(line[probabilityIndex]) >= 0 && Number(line[probabilityIndex]) <= 1);
        renderProfile(file, headers, data, tableRows, probabilityHeader, hasValidScores, hasTarget);
        if (hasValidScores && !hasTarget) {
          probabilityKey = probabilityHeader;
          rows = tableRows.map((row) => ({...row, [probabilityKey]: Number(row[headers[probabilityIndex]])}));
          source = 'upload';
          $('#queue-mode').textContent = 'UPLOADED SCORES';
          $('#queue-mode').className = 'demo-pill uploaded-pill';
          $('#queue-disclaimer').textContent = 'Priorities use only probability values already present in your file. The site does not generate or validate those estimates.';
          setStatus(`${file.name} loaded. Risk ranking uses the supplied “${headers[probabilityIndex]}” column only.`, 'success');
        } else {
          rows = [];
          source = 'unavailable';
          $('#queue-mode').textContent = 'INFERENCE UNAVAILABLE';
          $('#queue-mode').className = 'demo-pill unavailable-pill';
          $('#queue-disclaimer').textContent = hasTarget ? 'This labeled file can be inspected, but the review queue is intended for unlabeled rows.' : 'No risk scores are generated here. Use the local Streamlit research app for experimental Python scoring.';
          setStatus(`${file.name} inspected locally. No risk ranking was run.`, 'success');
        }
        renderQueue();
      } catch (error) {
        setError(`Could not parse this CSV: ${error.message}`);
      }
    };
    reader.readAsText(file);
  }
  function setStatus(message, state='') { fileStatus.textContent=message; fileStatus.className=`message ${state}`.trim(); }
  function setError(message) { setStatus(message,'error'); profile.hidden=true; renderQueue(); }
  function renderProfile(file, cols, data, tableRows, probKey, validScores, hasTarget) {
    profile.hidden=false;
    $('#filename').textContent=file.name;
    $('#stat-rows').textContent=data.length.toLocaleString();
    $('#stat-columns').textContent=cols.length.toLocaleString();
    const missing=data.reduce((sum,row)=>sum+cols.reduce((n,_,index)=>n+(String(row[index]??'').trim()===''?1:0),0),0);
    $('#stat-missing').textContent=missing.toLocaleString();
    const seen=new Set(); let duplicates=0;
    for (const row of data) { const key=JSON.stringify(row); if(seen.has(key))duplicates++; else seen.add(key); }
    $('#stat-duplicates').textContent=duplicates.toLocaleString();
    const probabilityBox=$('#probability-status');
    if(hasTarget) { probabilityBox.className='probability-state unavailable'; probabilityBox.textContent='A labeled target column is present. This queue is for unlabeled review prioritization, so risk ranking is disabled for this file.'; }
    else if(validScores) { probabilityBox.className='probability-state available'; probabilityBox.textContent=`Valid supplied scores found in “${cols[cols.findIndex((header)=>normalizeProbability(header))]}”. The queue uses these values without changing them.`; }
    else if(probKey) { probabilityBox.className='probability-state unavailable'; probabilityBox.textContent=`A “${cols[cols.findIndex((header)=>normalizeProbability(header))]}” column exists, but one or more values are blank or outside 0 to 1. Risk prioritization is unavailable for this file.`; }
    else { probabilityBox.className='probability-state unavailable'; probabilityBox.textContent='No supported probability column found. This page can inspect the file, but has no hosted model to score it.'; }
    $('#preview-head').innerHTML=`<tr>${cols.slice(0,6).map((column)=>`<th scope="col">${escapeHTML(column)}</th>`).join('')}</tr>`;
    $('#preview-body').innerHTML=data.slice(0,5).map((row)=>`<tr>${row.slice(0,6).map((cell)=>`<td title="${escapeHTML(cell)}">${escapeHTML(cell)||'Missing'}</td>`).join('')}</tr>`).join('');
  }
  function resetFile() {
    currentFile=null;fileInput.value='';profile.hidden=true;rows=demoRows.map((row)=>({...row}));headers=['order_id','region','category','return_probability'];probabilityKey='probability';source='demo';
    search.value='';queueSelect.value='25';customInput.value='25';currentCapacity=25;$('.custom-capacity[for="capacity-custom"]').hidden=true;customInput.hidden=true;
    $('#queue-mode').textContent='DEMO';$('#queue-mode').className='demo-pill';$('#queue-disclaimer').textContent='Demo rows are invented for interface preview. Uploaded rows are prioritized only when they contain a valid supplied probability.';setStatus('No file selected yet.');renderQueue();
  }
  function renderQueue() {
    const body=$('#queue-body'), head=$('#queue-head'), empty=$('#queue-empty');
    const query=search.value.trim().toLowerCase();
    const canRank=source==='demo'||source==='upload';
    if(!canRank) {
      head.innerHTML='';body.innerHTML='';empty.hidden=false;empty.textContent='Risk prioritization is unavailable because this static site has no inference service. Upload a file with a valid return probability to rank its rows.';
      $('#queue-total').textContent='0';$('#capacity-summary').textContent='No scored rows available';$('#queue-selection').textContent='No priority set available';$('#download-queue').disabled=true;return;
    }
    const sorted=[...rows].sort((a,b)=>Number(b[probabilityKey])-Number(a[probabilityKey]));
    const matching=sorted.filter((row)=>Object.values(row).some((value)=>String(value).toLowerCase().includes(query)));
    const selected=matching.slice(0,currentCapacity);
    const visibleColumns=source==='demo'?['order_id','region','category']:headers.slice(0,3);
    head.innerHTML=`<tr><th scope="col">RANK</th>${visibleColumns.map((column)=>`<th scope="col">${escapeHTML(column)}</th>`).join('')}<th scope="col">PROBABILITY</th><th scope="col">SET</th></tr>`;
    body.innerHTML=matching.slice(0,100).map((row,index)=>{
      const isPriority=index<currentCapacity;
      const identifier=visibleColumns.find((key)=>String(key).toLowerCase().includes('id'))||visibleColumns[0];
      const label=String(row[identifier]??`Row ${index+1}`);
      return `<tr><td>${index+1}</td>${visibleColumns.map((column)=>`<td title="${escapeHTML(row[column])}">${escapeHTML(row[column])||'Missing'}</td>`).join('')}<td class="probability">${(Number(row[probabilityKey])*100).toFixed(1)}%</td><td><span class="priority-label ${isPriority?'':'outside'}">${isPriority?'PRIORITY':'QUEUED'}</span></td></tr>`;
    }).join('');
    empty.hidden=matching.length!==0;
    if(!matching.length) empty.textContent='No rows match that search. Clear the search to see the queue.';
    $('#queue-total').textContent=rows.length.toLocaleString();
    $('#capacity-summary').textContent=`Showing first ${Math.min(100,matching.length).toLocaleString()} of ${matching.length.toLocaleString()} matching rows`;
    $('#queue-selection').textContent=`${selected.length.toLocaleString()} rows in the selected priority set${source==='demo'?' · illustrative demo':''}`;
    $('#download-queue').disabled=selected.length===0;
  }
  function csvCell(value) { const text=String(value??'');return /[",\r\n]/.test(text)?`"${text.replace(/"/g,'""')}"`:text; }
  function downloadQueue() {
    if(source!=='demo'&&source!=='upload') return;
    const query=search.value.trim().toLowerCase();
    const selected=[...rows].sort((a,b)=>Number(b[probabilityKey])-Number(a[probabilityKey])).filter((row)=>Object.values(row).some((value)=>String(value).toLowerCase().includes(query))).slice(0,currentCapacity);
    if(!selected.length)return;
    const dataHeaders=source==='demo'?['order_id','region','category','return_probability']:headers;
    const lines=[['review_rank',...dataHeaders].map(csvCell).join(','),...selected.map((row,index)=>[index+1,...dataHeaders.map((header)=>header===probabilityKey?row[probabilityKey]:row[header])].map(csvCell).join(','))];
    const blob=new Blob([lines.join('\r\n')],{type:'text/csv;charset=utf-8'});
    const url=URL.createObjectURL(blob);const anchor=document.createElement('a');anchor.href=url;anchor.download='returnwise-priority-queue.csv';document.body.append(anchor);anchor.click();anchor.remove();URL.revokeObjectURL(url);
  }
  renderQueue();
})();
