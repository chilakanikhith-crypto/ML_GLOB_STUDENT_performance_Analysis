// ============================================================
// ACADEMIC PERFORMANCE ANALYTICS
// Dashboard Data and Charts
// ============================================================

document.addEventListener("DOMContentLoaded", function () {
    loadDashboardData();
});


// ============================================================
// LOAD DASHBOARD DATA
// ============================================================

async function loadDashboardData() {
    try {
        const response = await fetch("/api/dashboard", {
            headers: {
                Accept: "application/json"
            }
        });

        if (!response.ok) {
            throw new Error(
                `Dashboard request failed: HTTP ${response.status}`
            );
        }

        const data = await response.json();

        if (!data.success) {
            throw new Error(
                data.error || "Unable to load dashboard data."
            );
        }

        updateKPIs(data.kpis || {});

        createGPAChart(data.gpa_distribution || {});
        createGradeChart(data.grade_distribution || {});
        createStudyGPAChart(data.study_time_gpa || {});
        createAbsenceGPAChart(data.absences_vs_gpa || {});
        createParentalSupportChart(
            data.parental_support_gpa || {}
        );
        createActivitiesChart(data.activity_data || {});

    } catch (error) {
        console.error("Failed to load dashboard:", error);

        showDashboardError(
            "Unable to load dashboard data. Check the Flask server and try refreshing."
        );
    }
}


// ============================================================
// SAFE ELEMENT UPDATE
// ============================================================

function setText(elementId, value, decimals = null) {
    const element = document.getElementById(elementId);

    if (!element) {
        return;
    }

    const number = Number(value);

    if (value === null || value === undefined || value === "" ||
        !Number.isFinite(number)) {
        element.textContent = "—";
        return;
    }

    element.textContent = decimals === null
        ? String(value)
        : number.toFixed(decimals);
}


// ============================================================
// KPI VALUES
// ============================================================

function updateKPIs(kpis) {
    setText("totalStudents", kpis.total_students);
    setText("averageGPA", kpis.average_gpa, 2);
    setText("highestGPA", kpis.highest_gpa, 2);
    setText("lowestGPA", kpis.lowest_gpa, 2);
    setText("averageStudyTime", kpis.average_study_time, 2);
    setText("averageAbsences", kpis.average_absences, 2);

    setText(
        "studentsNeedingImprovement",
        kpis.students_needing_improvement
    );
}


// ============================================================
// CHART HELPER
// ============================================================

function renderDashboardChart(canvasId, config) {
    const canvas = document.getElementById(canvasId);

    if (!canvas) {
        console.warn(`Chart canvas not found: ${canvasId}`);
        return null;
    }

    if (typeof Chart === "undefined") {
        console.error(
            "Chart.js is not loaded. Include Chart.js before dashboard.js."
        );
        return null;
    }

    // Prevent "Canvas is already in use" errors.
    const existingChart = Chart.getChart(canvas);

    if (existingChart) {
        existingChart.destroy();
    }

    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,

        animation: {
            duration: 650
        },

        plugins: {
            legend: {
                labels: {
                    usePointStyle: true,
                    boxWidth: 8,
                    padding: 16,
                    font: {
                        size: 11
                    }
                }
            },

            tooltip: {
                padding: 12,
                cornerRadius: 8
            }
        }
    };

    config.options = {
        ...commonOptions,
        ...config.options,
        plugins: {
            ...commonOptions.plugins,
            ...(config.options?.plugins || {})
        }
    };

    return new Chart(canvas, config);
}


// ============================================================
// GPA DISTRIBUTION
// ============================================================

function createGPAChart(data) {
    const labels = Object.keys(data);
    const values = Object.values(data).map(Number);

    if (!labels.length) {
        console.warn("No GPA distribution data available.");
        return;
    }

    renderDashboardChart("gpaDistributionChart", {
        type: "bar",

        data: {
            labels: labels,

            datasets: [{
                label: "Number of Students",
                data: values,

                backgroundColor: "#6366f1",
                hoverBackgroundColor: "#4338ca",
                borderRadius: 6,
                borderSkipped: false
            }]
        },

        options: {
            plugins: {
                legend: {
                    display: false
                }
            },

            scales: {
                x: {
                    title: {
                        display: true,
                        text: "GPA Range"
                    },

                    grid: {
                        display: false
                    }
                },

                y: {
                    beginAtZero: true,

                    title: {
                        display: true,
                        text: "Number of Students"
                    },

                    ticks: {
                        precision: 0
                    }
                }
            }
        }
    });
}


// ============================================================
// GRADE CLASS DISTRIBUTION
// ============================================================

function createGradeChart(data) {
    const classLabels = [
        "Excellent",
        "Very Good",
        "Good",
        "Needs Improvement",
        "At Risk"
    ];

    const labels = ["0", "1", "2", "3", "4"];

    const values = labels.map(function (gradeClass) {
        return Number(data[gradeClass] ?? data[Number(gradeClass)] ?? 0);
    });

    renderDashboardChart("gradeDistributionChart", {
        type: "doughnut",

        data: {
            labels: classLabels,

            datasets: [{
                label: "Students",
                data: values,

                backgroundColor: [
                    "#22c55e",
                    "#3b82f6",
                    "#f59e0b",
                    "#f97316",
                    "#ef4444"
                ],

                borderColor: "#ffffff",
                borderWidth: 3,
                hoverOffset: 8
            }]
        },

        options: {
            cutout: "62%",

            plugins: {
                legend: {
                    display: true,
                    position: "bottom"
                }
            }
        }
    });
}


// ============================================================
// STUDY TIME VS GPA
// ============================================================

function createStudyGPAChart(data) {
    if (
        !Array.isArray(data.labels) ||
        !Array.isArray(data.values)
    ) {
        console.warn("Invalid study time vs GPA data.");
        return;
    }

    renderDashboardChart("studyGPAChart", {
        type: "line",

        data: {
            labels: data.labels,

            datasets: [{
                label: "Average GPA",
                data: data.values,

                borderColor: "#4f46e5",
                backgroundColor: "rgba(79, 70, 229, 0.12)",

                borderWidth: 3,
                pointRadius: 4,
                pointHoverRadius: 7,
                pointBackgroundColor: "#4f46e5",

                tension: 0.35,
                fill: true
            }]
        },

        options: {
            scales: {
                x: {
                    title: {
                        display: true,
                        text: "Study Time (hours/week)"
                    },

                    grid: {
                        display: false
                    }
                },

                y: {
                    beginAtZero: true,
                    min: 1,
                    max: 10,

                    title: {
                        display: true,
                        text: "Average GPA (1–10)"
                    }
                }
            }
        }
    });
}


// ============================================================
// ABSENCES VS GPA
// ============================================================

function createAbsenceGPAChart(data) {
    if (
        !Array.isArray(data.x) ||
        !Array.isArray(data.y)
    ) {
        console.warn("Invalid absences vs GPA data.");
        return;
    }

    const points = data.x.map(function (x, index) {
        return {
            x: Number(x),
            y: Number(data.y[index])
        };
    }).filter(function (point) {
        return Number.isFinite(point.x) &&
               Number.isFinite(point.y);
    });

    renderDashboardChart("absenceGPAChart", {
        type: "scatter",

        data: {
            datasets: [{
                label: "Students",
                data: points,

                backgroundColor: "rgba(6, 182, 212, 0.65)",
                borderColor: "#0891b2",

                pointRadius: 4,
                pointHoverRadius: 6
            }]
        },

        options: {
            plugins: {
                legend: {
                    display: false
                }
            },

            scales: {
                x: {
                    type: "linear",

                    title: {
                        display: true,
                        text: "Number of Absences"
                    }
                },

                y: {
                    min: 1,
                    max: 10,

                    title: {
                        display: true,
                        text: "GPA (1–10)"
                    }
                }
            }
        }
    });
}


// ============================================================
// PARENTAL SUPPORT VS GPA
// ============================================================

function createParentalSupportChart(data) {
    if (
        !Array.isArray(data.labels) ||
        !Array.isArray(data.values)
    ) {
        console.warn("Invalid parental support data.");
        return;
    }

    renderDashboardChart("parentalSupportChart", {
        type: "bar",

        data: {
            labels: data.labels,

            datasets: [{
                label: "Average GPA",
                data: data.values,

                backgroundColor: [
                    "#818cf8",
                    "#6366f1",
                    "#4f46e5",
                    "#4338ca",
                    "#3730a3"
                ],

                borderRadius: 6,
                borderSkipped: false
            }]
        },

        options: {
            plugins: {
                legend: {
                    display: false
                }
            },

            scales: {
                x: {
                    title: {
                        display: true,
                        text: "Parental Support Level"
                    },

                    grid: {
                        display: false
                    }
                },

                y: {
                    min: 1,
                    max: 10,

                    title: {
                        display: true,
                        text: "Average GPA (1–10)"
                    }
                }
            }
        }
    });
}


// ============================================================
// EXTRACURRICULAR ACTIVITIES VS GPA
// ============================================================

function createActivitiesChart(data) {
    const activities = Object.keys(data);

    if (!activities.length) {
        console.warn("No activity data available.");
        return;
    }

    const labels = ["No", "Yes"];

    const datasets = activities.map(function (activity, index) {
        const values = data[activity] || {};

        const activityLabels = values.labels || [];
        const activityValues = values.values || [];

        return {
            label: activity,

            data: labels.map(function (label) {
                const position = activityLabels.indexOf(label);

                return position === -1
                    ? null
                    : Number(activityValues[position]);
            }),

            backgroundColor: [
                "#6366f1",
                "#06b6d4",
                "#22c55e",
                "#f59e0b",
                "#8b5cf6",
                "#ec4899"
            ][index % 6],

            borderRadius: 5,
            borderSkipped: false
        };
    });

    renderDashboardChart("activitiesChart", {
        type: "bar",

        data: {
            labels: labels,
            datasets: datasets
        },

        options: {
            scales: {
                x: {
                    title: {
                        display: true,
                        text: "Activity Participation"
                    },

                    grid: {
                        display: false
                    }
                },

                y: {
                    min: 1,
                    max: 10,

                    title: {
                        display: true,
                        text: "Average GPA (1–10)"
                    }
                }
            }
        }
    });
}


// ============================================================
// DASHBOARD ERROR MESSAGE
// ============================================================

function showDashboardError(message) {
    const container = document.getElementById(
        "dashboardError"
    );

    if (container) {
        container.textContent = message;
        container.hidden = false;
    }
}