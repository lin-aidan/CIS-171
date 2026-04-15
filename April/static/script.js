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
});