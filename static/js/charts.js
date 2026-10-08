function createChart(canvasId, type, labels, data, label) {

    const canvas = document.getElementById(canvasId);

    if (!canvas) {
        return;
    }

    new Chart(canvas, {
        type: type,

        data: {
            labels: labels,

            datasets: [{
                label: label,
                data: data,
                borderWidth: 2,
                tension: 0.3
            }]
        },

        options: {
            responsive: true,
            maintainAspectRatio: false,

            plugins: {
                legend: {
                    display: true
                }
            },

            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}