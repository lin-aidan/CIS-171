document.addEventListener('DOMContentLoaded', function() {
    fetch('/vocab')
        .then(response => response.json())
        .then(data => {
            const container = document.getElementById('vocab-container');
            for (const [word, index] of Object.entries(data)) {
                const card = document.createElement('div');
                card.className = 'vocab-card';
                card.innerHTML = `
                    <div class="word">${word}</div>
                    <div class="arrow">→</div>
                    <div class="index">${index}</div>
                `;
                container.appendChild(card);
            }
        })
        .catch(error => console.error('Error fetching vocab:', error));
    
    // Handle encode form submission
    const form = document.getElementById('encode-form');
    const input = document.getElementById('text-input');
    const resultDiv = document.getElementById('encoding-result');

    form.addEventListener('submit', function(e) {
        e.preventDefault();
        const text = input.value || '';
        fetch(`/encode?text=${encodeURIComponent(text)}`)
            .then(res => res.json())
            .then(payload => {
                const enc = payload.encoding || [];
                // display as small chips
                resultDiv.innerHTML = '';
                const title = document.createElement('div');
                title.textContent = `Encoding for: "${payload.text}"`;
                title.style.fontWeight = 'bold';
                title.style.marginBottom = '6px';
                resultDiv.appendChild(title);

                const chips = document.createElement('div');
                chips.style.display = 'flex';
                chips.style.gap = '6px';
                enc.forEach(n => {
                    const c = document.createElement('div');
                    c.className = 'chip';
                    c.textContent = n;
                    chips.appendChild(c);
                });
                resultDiv.appendChild(chips);
            })
            .catch(err => {
                resultDiv.textContent = 'Error encoding text.';
                console.error(err);
            });
    });
});