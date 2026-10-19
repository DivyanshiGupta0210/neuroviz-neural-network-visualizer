const $ = (id) => document.getElementById(id);
const state = {
  inputCount: 2,
  hiddenSizes: [3],
  activation: "sigmoid",
  inputs: [0.7, 0.2, 0.5, 1],
  biases: {},
  weights: {},
  selected: { type: "neuron", layer: 0, index: 0 }
};
const MAX_LAYERS = 3, MAX_NEURONS = 5;
const layerSizes = () => [state.inputCount, ...state.hiddenSizes, 1];
const keyNeuron = (layer, index) => `${layer}:${index}`;
const keyWeight = (fromLayer, fromIndex, toIndex) => `${fromLayer}:${fromIndex}>${toIndex}`;
const fmt = (n) => Number(n).toFixed(3).replace(/\.?0+$/, "");
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

function initParameters() {
  const sizes = layerSizes();
  for (let l = 1; l < sizes.length; l++) {
    for (let j = 0; j < sizes[l]; j++) {
      const nk = keyNeuron(l, j);
      if (state.biases[nk] === undefined) state.biases[nk] = 0;
      for (let i = 0; i < sizes[l - 1]; i++) {
        const wk = keyWeight(l - 1, i, j);
        if (state.weights[wk] === undefined) {
          state.weights[wk] = Number((((i + 1) * (j + 2) + l) % 7 - 3) / 4).toFixed(2) * 1;
        }
      }
    }
  }
}
function activation(x) {
  if (state.activation === "relu") return Math.max(0, x);
  if (state.activation === "tanh") return Math.tanh(x);
  return 1 / (1 + Math.exp(-clamp(x, -30, 30)));
}
function activationLabel() {
  return state.activation === "relu" ? "ReLU" : state.activation === "tanh" ? "tanh" : "sigmoid";
}
function calculate() {
  initParameters();
  const sizes = layerSizes();
  const values = [state.inputs.slice(0, state.inputCount)];
  const steps = [];
  for (let l = 1; l < sizes.length; l++) {
    const layerValues = [];
    for (let j = 0; j < sizes[l]; j++) {
      const terms = [];
      let total = Number(state.biases[keyNeuron(l, j)] || 0);
      for (let i = 0; i < sizes[l - 1]; i++) {
        const input = values[l - 1][i];
        const weight = Number(state.weights[keyWeight(l - 1, i, j)] || 0);
        terms.push({ input, weight, from: i });
        total += input * weight;
      }
      const out = activation(total);
      layerValues.push(out);
      steps.push({
        layer: l, neuron: j, terms, bias: Number(state.biases[keyNeuron(l, j)] || 0),
        total, output: out, fn: activationLabel()
      });
    }
    values.push(layerValues);
  }
  return { values, steps, output: values[values.length - 1][0] };
}
function syncLayerControls() {
  const host = $("layerControls");
  host.innerHTML = "";
  state.hiddenSizes.forEach((size, idx) => {
    const section = document.createElement("section");
    section.className = "layer-control";
    section.innerHTML = `
      <div class="control-heading"><label for="hidden-${idx}">Neurons · hidden layer ${idx + 1}</label><output id="hidden-out-${idx}">${size}</output></div>
      <input id="hidden-${idx}" type="range" min="1" max="${MAX_NEURONS}" step="1" value="${size}">
      <div class="range-labels"><span>1</span><span>${MAX_NEURONS}</span></div>`;
    host.appendChild(section);
    section.querySelector("input").addEventListener("input", e => {
      state.hiddenSizes[idx] = Number(e.target.value);
      $(`hidden-out-${idx}`).value = e.target.value;
      cleanSelection(); render();
    });
  });
}
function cleanSelection() {
  const sizes = layerSizes(), s = state.selected;
  if (s.type === "neuron") {
    if (s.layer >= sizes.length) state.selected = {type:"neuron",layer:0,index:0};
    else if (s.index >= sizes[s.layer]) s.index = sizes[s.layer] - 1;
  } else {
    if (s.layer >= sizes.length - 1 || s.from >= sizes[s.layer] || s.to >= sizes[s.layer+1]) {
      state.selected = {type:"neuron",layer:0,index:0};
    }
  }
}
function renderNetwork(result) {
  const sizes = layerSizes(), width = Math.max(570, sizes.length * 150), height = 300;
  const left = 52, right = width - 52, xGap = (right-left)/(sizes.length-1 || 1);
  const coords = [];
  sizes.forEach((count,l) => {
    coords[l] = [];
    const usable = Math.min(210, height-65);
    const gap = count === 1 ? 0 : usable/(count-1);
    for(let i=0;i<count;i++) coords[l].push({x:left+l*xGap,y:height/2-(count-1)*gap/2+i*gap});
  });
  let svg = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Neural network. Select a neuron or connection.">`;
  // connections behind neurons
  for(let l=0;l<sizes.length-1;l++){
    for(let i=0;i<sizes[l];i++) for(let j=0;j<sizes[l+1];j++){
      const a=coords[l][i],b=coords[l+1][j],w=Number(state.weights[keyWeight(l,i,j)]||0);
      const selected=state.selected.type==="weight"&&state.selected.layer===l&&state.selected.from===i&&state.selected.to===j;
      const color=w<0?"#D28A75":"#72A98D", stroke=selected?"#263D32":color, sw=selected?3.5:1+Math.min(2,Math.abs(w));
      svg+=`<line class="weight-line" data-layer="${l}" data-from="${i}" data-to="${j}" x1="${a.x+17}" y1="${a.y}" x2="${b.x-17}" y2="${b.y}" stroke="${stroke}" stroke-width="${sw}" opacity="${selected?1:.68}"><title>Weight ${fmt(w)}. Click to edit.</title></line>`;
      svg+=`<text x="${(a.x+b.x)/2}" y="${(a.y+b.y)/2-4}" font-size="10" text-anchor="middle" fill="${w<0?"#B16F5E":"#54876D"}" pointer-events="none">${fmt(w)}</text>`;
    }
  }
  // layer labels
  sizes.forEach((count,l)=>{
    const label=l===0?"INPUT":l===sizes.length-1?"OUTPUT":`HIDDEN ${l}`;
    svg+=`<text x="${coords[l][0].x}" y="19" text-anchor="middle" fill="#7A897F" font-size="10" font-weight="700" letter-spacing="1">${label}</text>`;
    coords[l].forEach((p,i)=>{
      const selected=state.selected.type==="neuron"&&state.selected.layer===l&&state.selected.index===i;
      const fill=l===0?"#E3EFFB":l===sizes.length-1?"#DDF1E5":"#F0E9FA";
      const stroke=l===0?"#6C9BCF":l===sizes.length-1?"#5A9A7B":"#9B80C7";
      const value=result.values[l][i];
      svg+=`<g class="neuron" data-layer="${l}" data-index="${i}"><circle cx="${p.x}" cy="${p.y}" r="17" fill="${fill}" stroke="${selected?"#20392F":stroke}" stroke-width="${selected?3:1.5}"/><text x="${p.x}" y="${p.y+3.5}" text-anchor="middle" font-size="10" font-weight="700" fill="#29463A" pointer-events="none">${fmt(value)}</text><title>Layer ${l}, neuron ${i+1}: ${fmt(value)}. Click to inspect.</title></g>`;
    });
  });
  svg += `</svg>`;
  $("networkCanvas").innerHTML=svg;
  $("networkCanvas").querySelectorAll(".neuron").forEach(el=>el.addEventListener("click",()=>{
    state.selected={type:"neuron",layer:Number(el.dataset.layer),index:Number(el.dataset.index)}; render();
  }));
  $("networkCanvas").querySelectorAll(".weight-line").forEach(el=>el.addEventListener("click",()=>{
    state.selected={type:"weight",layer:Number(el.dataset.layer),from:Number(el.dataset.from),to:Number(el.dataset.to)}; render();
  }));
}
function renderEditor(result) {
  const s=state.selected, host=$("editorFields");
  host.innerHTML="";
  const addField=(label,value,help,onchange,step="0.1")=>{
    const wrap=document.createElement("div");wrap.className="field";
    wrap.innerHTML=`<label>${label}</label><input type="number" step="${step}" value="${value}"><small>${help}</small>`;
    wrap.querySelector("input").addEventListener("change",e=>{const v=Number(e.target.value);if(Number.isFinite(v)){onchange(v);render();}});
    host.appendChild(wrap);
  };
  if(s.type==="weight"){
    $("selectedTitle").textContent=`Weight · neuron ${s.from+1} → ${s.to+1}`;
    $("selectedType").textContent="CONNECTION";
    const k=keyWeight(s.layer,s.from,s.to);
    addField("Connection weight",state.weights[k],"How strongly this connection influences the next neuron.",v=>state.weights[k]=v);
    addField("From neuron value",fmt(result.values[s.layer][s.from]),"Calculated by the previous layer.",()=>{},"0.001");
    addField("Contribution",fmt(result.values[s.layer][s.from]*state.weights[k]),"Input value × weight.",()=>{},"0.001");
  }else if(s.layer===0){
    $("selectedTitle").textContent=`Input neuron ${s.index+1}`;
    $("selectedType").textContent="INPUT";
    addField("Input value",state.inputs[s.index],"The number you feed into the network.",v=>state.inputs[s.index]=v);
    addField("Current value",fmt(result.values[0][s.index]),"This value is passed to the next layer.",()=>{},"0.001");
    const outgoing=state.hiddenSizes.length?state.weights[keyWeight(0,s.index,0)]:state.weights[keyWeight(0,s.index,0)];
    addField("First connection",outgoing,"Select a line in the diagram to edit its weight.",()=>{},"0.001");
  }else{
    $("selectedTitle").textContent=`${s.layer===layerSizes().length-1?"Output":"Hidden layer "+s.layer} · neuron ${s.index+1}`;
    $("selectedType").textContent=s.layer===layerSizes().length-1?"OUTPUT":"NEURON";
    const k=keyNeuron(s.layer,s.index);
    addField("Bias",state.biases[k]||0,"A small adjustable offset added before activation.",v=>state.biases[k]=v);
    addField("Weighted total",fmt(result.steps.find(x=>x.layer===s.layer&&x.neuron===s.index).total),"Sum of incoming (value × weight) plus bias.",()=>{},"0.001");
    addField("Activation",fmt(result.values[s.layer][s.index]),`${activationLabel()} applied to the weighted total.`,()=>{},"0.001");
  }
}
function renderTeacher(result) {
  const host=$("lessonSteps");host.innerHTML="";
  let n=0;
  result.steps.forEach(step=>{
    n++;
    const terms=step.terms.map(t=>`(${fmt(t.input)} × ${fmt(t.weight)})`).join(" + ");
    const formula=`${terms}${step.bias>=0?" + ":" − "}${fmt(Math.abs(step.bias))}`;
    const div=document.createElement("div");div.className="lesson-step";
    div.innerHTML=`<div class="step-number">${n}</div><div class="step-copy"><strong>Layer ${step.layer}, neuron ${step.neuron+1}: calculate the total</strong><p>We multiply each incoming value by its connection weight, add the results and then add the bias.</p><p>Next, we apply ${step.fn} to turn the total into this neuron's output.</p></div><div class="formula">${formula} = ${fmt(step.total)}<br>${step.fn}(${fmt(step.total)}) = ${fmt(step.output)}</div>`;
    host.appendChild(div);
  });
  const final=result.output;
  $("finalOutput").textContent=fmt(final);
}
function render() {
  initParameters();
  const result=calculate();
  $("inputCountOut").value=state.inputCount;
  $("hiddenCountOut").value=state.hiddenSizes.length;
  renderNetwork(result);renderEditor(result);renderTeacher(result);
}
$("inputCount").addEventListener("input",e=>{
  const old=state.inputCount;state.inputCount=Number(e.target.value);
  for(let i=old;i<state.inputCount;i++) if(state.inputs[i]===undefined)state.inputs[i]=0.5;
  cleanSelection();render();
});
$("hiddenCount").addEventListener("input",e=>{
  const count=Number(e.target.value);
  while(state.hiddenSizes.length<count)state.hiddenSizes.push(3);
  state.hiddenSizes=state.hiddenSizes.slice(0,count);
  syncLayerControls();cleanSelection();render();
});
$("activation").addEventListener("change",e=>{state.activation=e.target.value;render();});
$("resetBtn").addEventListener("click",()=>{
  state.inputs=[0.7,0.2,0.5,1];state.biases={};state.weights={};state.activation="sigmoid";
  $("activation").value="sigmoid";state.inputCount=2;state.hiddenSizes=[3];
  $("inputCount").value=2;$("hiddenCount").value=1;syncLayerControls();
  state.selected={type:"neuron",layer:0,index:0};render();
});
syncLayerControls();render();
