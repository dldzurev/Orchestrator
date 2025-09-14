(function(){
  // ---- small utilities ----
  function onReady(fn){
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', fn, { once: true });
    } else {
      fn();
    }
  }

  // Find the Strategy screen element across common ids/classes
  function findStrategyRoot(){
    const candidates = [
      '#strategyScreen',        // current layout
      '#strategy-content',      // older layout
      '#strategy', '#strategyPage'
    ];
    for (const sel of candidates) {
      const el = document.querySelector(sel);
      if (el) return el;
    }
    // fallback: any element whose id contains 'strategy' or placeholder text
    const all = Array.from(document.querySelectorAll('.screen,.view-content,div,section'));
    for (const el of all) {
      const id = (el.id || '').toLowerCase();
      const txt = (el.textContent || '').toLowerCase();
      if (id.includes('strategy')) return el;
      if (txt.includes('strategy builder') && txt.includes('coming soon')) return el;
    }
    return null;
  }

  function ensureOnce(root){
    return root && !root.querySelector('#strategyApp');
  }

  function mountStrategyUI(root){
    if (!ensureOnce(root)) return;

    root.innerHTML = `
      <div class="strategy-container" id="strategyApp">
        ${cardHTML('Commodity', 'commodity')}
        ${cardHTML('Buy', 'buy')}
        ${cardHTML('Sell', 'sell')}
      </div>
      <datalist id="qtySuggestions">
        <option value="1"></option>
        <option value="5"></option>
        <option value="10"></option>
      </datalist>
    `;

    ['commodity','buy','sell'].forEach(type => initCard(type));
  }

  // Re-mount when user clicks the Strategy tab (works with your nav buttons)
  function hookNavForRemount(){
    document.querySelectorAll('.nav-btn[data-view]').forEach(btn=>{
      btn.addEventListener('click', () => {
        const view = btn.getAttribute('data-view');
        if (String(view).toLowerCase() === 'strategy') {
          const root = findStrategyRoot();
          if (root) mountStrategyUI(root);
        }
      });
    });
  }

  // MutationObserver: if the Strategy container is swapped in later, mount then
  function observeForStrategy(){
    const obs = new MutationObserver(() => {
      const root = findStrategyRoot();
      if (root && ensureOnce(root)) {
        mountStrategyUI(root);
      }
    });
    obs.observe(document.documentElement, { childList:true, subtree:true });
  }

  // ---------- Cards ----------
  function cardHTML(title, type){
    return `
      <div class="str-card" data-card="${type}">
        <div class="str-card__header">${title}</div>
        <div class="str-card__body">
          <div class="str-rows"></div>
          <button class="str-add" type="button">(+)&nbsp;Add</button>
          <div class="str-help">${type === 'commodity'
            ? 'Add one or more items.'
            : 'Select an item, then enter a number.'}
          </div>
        </div>
      </div>
    `;
  }

  function initCard(type){
    const card = document.querySelector(`.str-card[data-card="${type}"]`);
    if (!card) return;
    const rowsEl = card.querySelector('.str-rows');
    const addBtn = card.querySelector('.str-add');

    // start with one row
    addRow(type, rowsEl);

    // add more rows
    addBtn.addEventListener('click', () => addRow(type, rowsEl));

    // change: transform dropdown to split (buy/sell only)
    card.addEventListener('change', (e) => {
      const target = e.target;
      if (!target.classList.contains('str-select')) return;
      const row = target.closest('.str-row');
      const value = target.value;
      const rowType = row?.dataset?.type;
      if (!value) return;
      if (rowType === 'buy' || rowType === 'sell') {
        toSplitRow(row, value, rowType);
      }
    });

    // click left pill to re-edit the left side
    card.addEventListener('click', (e) => {
      const pill = e.target.closest('.str-pill');
      if (!pill) return;
      const row = pill.closest('.str-row');
      const rowType = row?.dataset?.type;
      const current = pill.dataset.value || 'a';
      row.innerHTML = '';
      row.appendChild(buildSelect(rowType, current));
    });
  }

  function addRow(type, rowsEl){
    const row = document.createElement('div');
    row.className = 'str-row';
    row.dataset.type = type;
    row.appendChild(buildSelect(type));
    rowsEl.appendChild(row);
  }

  function buildSelect(type, preselect=null){
    const sel = document.createElement('select');
    sel.className = 'str-select';
    sel.innerHTML = `
      <option value="" ${preselect ? '' : 'selected'} disabled>Select…</option>
      <option value="a" ${preselect==='a'?'selected':''}>a</option>
      <option value="b" ${preselect==='b'?'selected':''}>b</option>
      <option value="c" ${preselect==='c'?'selected':''}>c</option>
    `;
    return sel;
  }

  function toSplitRow(row, value, rowType){
    row.innerHTML = `
      <div class="str-split">
        <button type="button" class="str-pill" data-value="${value}" aria-label="${rowType} selection">${value}</button>
        <input class="str-number" type="number" placeholder="Enter number" min="0" step="1" list="qtySuggestions" />
      </div>
    `;
  }

  // ---- boot ----
  onReady(() => {
    hookNavForRemount();
    observeForStrategy();

    // Try immediate mount if the Strategy screen is already in DOM
    const rootNow = findStrategyRoot();
    if (rootNow) mountStrategyUI(rootNow);

    // Belt-and-suspenders: retry a few times in case of late DOM changes
    let tries = 0;
    const t = setInterval(() => {
      const r = findStrategyRoot();
      if (r) {
        mountStrategyUI(r);
        clearInterval(t);
      }
      if (++tries > 50) clearInterval(t);
    }, 100);
  });
})();
