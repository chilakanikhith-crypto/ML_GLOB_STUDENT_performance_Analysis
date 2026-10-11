// ============================================================
// ACADEMIC PERFORMANCE ANALYTICS
// Reusable Chart.js Helper
// ============================================================

function createChart(canvasId, type, labels, data, label) {
    const canvas = document.getElementById(canvasId);

    if (!canvas) {
        console.warn(`Chart canvas not found: ${canvasId}`);
        return null;
    }

    if (typeof Chart === "undefined") {
        console.error(
            "Chart.js is not loaded. Include Chart.js before charts.js."
        );
        return null;
    }

    // Destroy an existing chart on the same canvas before recreating it.
    const existingChart = Chart.getChart(canvas);

    if (existingChart) {
        existingChart.destroy();
    }

    const colors = [
        "#4f46e5",
        "#06b6d4",
        "#16a34a",
        "#f59e0b",
        "#ef4444",
        "#8b5cf6",
        "#ec4899",
        "#64748b"
    ];

    const isCircular = ["pie", "doughnut", "polarArea"].includes(type);
    const isLine = type === "line";

    const dataset = {
        label: label,
        data: data,
        borderWidth: isCircular ? 2 : 1,
        borderRadius: type === "bar" ? 6 : 0,
        backgroundColor: isCircular
            ? colors.slice(0, data.length)
            : isLine
                ? "rgba(79, 70, 229, 0.12)"
                : "#4f46e5",
        borderColor: isCircular
            ? "#ffffff"
            : isLine
                ? "#4f46e5"
                : "#4338ca"
    };

    if (isLine) {
        dataset.tension = 0.35;
        dataset.fill = true;
        dataset.pointRadius = 4;
        dataset.pointHoverRadius = 6;
        dataset.pointBackgroundColor = "#4f46e5";
    }

    if (type === "bar") {
        dataset.hoverBackgroundColor = "#3730a3";
    }

    const config = {
        type: type,

        data: {
            labels: labels,
            datasets: [dataset]
        },

        options: {
            responsive: true,
            maintainAspectRatio: false,

            interaction: {
                intersect: false,
                mode: "index"
            },

            animation: {
                duration: 700
            },

            plugins: {
                legend: {
                    display: true,
                    position: isCircular ? "bottom" : "top",
                    labels: {
                        usePointStyle: true,
                        padding: 16,
                        boxWidth: 8,
                        font: {
                            size: 11
                        }
                    }
                },

                tooltip: {
                    enabled: true,
                    padding: 12,
                    cornerRadius: 8,
                    displayColors: true
                }
            }
        }
    };

    // Cartesian charts use axes; circular charts do not.
    if (!isCircular) {
        config.options.scales = {
            x: {
                grid: {
                    display: false
                },
                ticks: {
                    color: "#6b7280",
                    font: {
                        size: 11
                    }
                },
                border: {
                    display: false
                }
            },

            y: {
                beginAtZero: true,
                grid: {
                    color: "rgba(148, 163, 184, 0.15)"
                },
                ticks: {
                    precision: 0,
                    color: "#6b7280",
                    font: {
                        size: 11
                    }
                },
                border: {
                    display: false
                }
            }
        };
    }

    return new Chart(canvas, config);
}


// ============================================================
// OPTIONAL: DESTROY A CHART
// ============================================================

function destroyChart(canvasId) {
    const canvas = document.getElementById(canvasId);

    if (!canvas || typeof Chart === "undefined") {
        return;
    }

    const chart = Chart.getChart(canvas);

    if (chart) {
        chart.destroy();
    }
}


// ============================================================
// OPTIONAL: CREATE A GRADE DISTRIBUTION DOUGHNUT CHART
// ============================================================

function createGradeDistributionChart(canvasId, labels, data) {
    return createChart(
        canvasId,
        "doughnut",
        labels,
        data,
        "Students"
    );
}


// ============================================================
// OPTIONAL: CREATE A GPA BAR CHART
// ============================================================

function createGPAChart(canvasId, labels, data) {
    return createChart(
        canvasId,
        "bar",
        labels,
        data,
        "Average GPA"
    );
}