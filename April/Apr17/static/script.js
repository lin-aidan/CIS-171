const apiUrl = '/api/vocab';

let vocab = {};
const select = document.getElementById('charSelect');
const tbody = document.querySelector('#lookupTable tbody');
const ctx = document.getElementById('embedChart').getContext('2d');
let chart = null;

function populateSelect() {
	const chars = Object.keys(vocab).sort((a,b)=>vocab[a].index - vocab[b].index);
	chars.forEach(c => {
		const opt = document.createElement('option');
		opt.value = c;
		opt.textContent = `${c} (${vocab[c].index})`;
		select.appendChild(opt);
	});
}

function renderTable() {
	tbody.innerHTML = '';
	const chars = Object.keys(vocab).sort((a,b)=>vocab[a].index - vocab[b].index);
	chars.forEach(c => {
		const row = document.createElement('tr');
		const tdChar = document.createElement('td'); tdChar.textContent = c;
		const tdIndex = document.createElement('td'); tdIndex.textContent = vocab[c].index;
		const tdEmbed = document.createElement('td'); tdEmbed.textContent = '[' + vocab[c].embedding.map(v=>v.toFixed(3)).join(', ') + ']';
		row.appendChild(tdChar); row.appendChild(tdIndex); row.appendChild(tdEmbed);
		tbody.appendChild(row);
	});
}

function plotEmbedding(embedding, label) {
	const labels = embedding.map((_,i)=>`d${i+1}`);
	const data = {
		labels,
		datasets: [{
			label: `Embedding: ${label}`,
			backgroundColor: 'rgba(54,162,235,0.2)',
			borderColor: 'rgba(54,162,235,1)',
			borderWidth: 1,
			data: embedding
		}]
	};

	if (chart) {
		chart.data = data;
		chart.update();
	} else {
		chart = new Chart(ctx, {
			type: 'bar',
			data,
			options: {scales: {y: {beginAtZero: false}}}
		});
	}
}

async function init() {
	try {
		const res = await fetch(apiUrl);
		vocab = await res.json();
		populateSelect();
		renderTable();
		// select first item
		if (select.options.length) {
			select.selectedIndex = 0;
			const first = select.value;
			plotEmbedding(vocab[first].embedding, first);
		}
	} catch (err) {
		console.error('Failed fetching vocab:', err);
	}
}

select.addEventListener('change', () => {
	const c = select.value;
	if (vocab[c]) plotEmbedding(vocab[c].embedding, c);
});

init();