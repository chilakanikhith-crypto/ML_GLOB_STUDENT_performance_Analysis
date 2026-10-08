document.addEventListener("DOMContentLoaded", function () {

    loadDashboardData();

});


async function loadDashboardData() {

    try {

        const response = await fetch("/api/dashboard");

        const data = await response.json();

        if (!data.success) {

            console.error(
                "Dashboard error:",
                data.error
            );

            return;
        }

        updateKPIs(data.kpis);

        createGPAChart(
            data.gpa_distribution
        );

        createGradeChart(
            data.grade_distribution
        );

        createStudyGPAChart(
            data.study_time_gpa
        );

        createAbsenceGPAChart(
            data.absences_vs_gpa
        );

        createParentalSupportChart(
            data.parental_support_gpa
        );

        createActivitiesChart(
            data.activity_data
        );

    } catch (error) {

        console.error(
            "Failed to load dashboard:",
            error
        );

    }

}


/* =====================================================
   KPI VALUES
   ===================================================== */

function updateKPIs(kpis) {

    document.getElementById(
        "totalStudents"
    ).textContent =
        kpis.total_students;


    document.getElementById(
        "averageGPA"
    ).textContent =
        kpis.average_gpa.toFixed(2);


    document.getElementById(
        "highestGPA"
    ).textContent =
        kpis.highest_gpa.toFixed(2);


    document.getElementById(
        "lowestGPA"
    ).textContent =
        kpis.lowest_gpa.toFixed(2);


    document.getElementById(
        "averageStudyTime"
    ).textContent =
        kpis.average_study_time.toFixed(2);


    document.getElementById(
        "averageAbsences"
    ).textContent =
        kpis.average_absences.toFixed(2);


    document.getElementById(
        "studentsNeedingImprovement"
    ).textContent =
        kpis.students_needing_improvement;

}


/* =====================================================
   GPA DISTRIBUTION
   ===================================================== */

function createGPAChart(data) {

    const canvas =
        document.getElementById(
            "gpaDistributionChart"
        );

    if (!canvas) return;

    new Chart(canvas, {

        type: "bar",

        data: {

            labels: Object.keys(data),

            datasets: [{

                label: "Number of Students",

                data: Object.values(data),

                borderWidth: 1

            }]

        },

        options: {

            responsive: true,

            maintainAspectRatio: false,

            plugins: {

                legend: {
                    display: false
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


/* =====================================================
   GRADE CLASS DISTRIBUTION
   ===================================================== */

function createGradeChart(data) {

    const canvas =
        document.getElementById(
            "gradeDistributionChart"
        );

    if (!canvas) return;

    const labels = [
        "Class 0",
        "Class 1",
        "Class 2",
        "Class 3",
        "Class 4"
    ];

    const values = labels.map(
        function (_, index) {
            return data[String(index)] || 0;
        }
    );


    new Chart(canvas, {

        type: "doughnut",

        data: {

            labels: labels,

            datasets: [{

                label: "Students",

                data: values,

                borderWidth: 2

            }]

        },

        options: {

            responsive: true,

            maintainAspectRatio: false

        }

    });

}


/* =====================================================
   STUDY TIME VS GPA
   ===================================================== */

function createStudyGPAChart(data) {

    const canvas =
        document.getElementById(
            "studyGPAChart"
        );

    if (!canvas) return;


    new Chart(canvas, {

        type: "line",

        data: {

            labels: data.labels,

            datasets: [{

                label: "Average GPA",

                data: data.values,

                borderWidth: 2,

                tension: 0.3,

                fill: false

            }]

        },

        options: {

            responsive: true,

            maintainAspectRatio: false,

            scales: {

                x: {

                    title: {
                        display: true,
                        text: "Study Time (hours/week)"
                    }

                },

                y: {

                    beginAtZero: true,

                    max: 4,

                    title: {
                        display: true,
                        text: "Average GPA"
                    }

                }

            }

        }

    });

}


/* =====================================================
   ABSENCES VS GPA
   ===================================================== */

function createAbsenceGPAChart(data) {

    const canvas =
        document.getElementById(
            "absenceGPAChart"
        );

    if (!canvas) return;


    const points = data.x.map(
        function (x, index) {

            return {
                x: x,
                y: data.y[index]
            };

        }
    );


    new Chart(canvas, {

        type: "scatter",

        data: {

            datasets: [{

                label: "Students",

                data: points,

                borderWidth: 1,

                pointRadius: 3

            }]

        },

        options: {

            responsive: true,

            maintainAspectRatio: false,

            scales: {

                x: {

                    title: {
                        display: true,
                        text: "Absences"
                    }

                },

                y: {

                    beginAtZero: true,

                    max: 4,

                    title: {
                        display: true,
                        text: "GPA"
                    }

                }

            }

        }

    });

}


/* =====================================================
   PARENTAL SUPPORT VS GPA
   ===================================================== */

function createParentalSupportChart(data) {

    const canvas =
        document.getElementById(
            "parentalSupportChart"
        );

    if (!canvas) return;


    new Chart(canvas, {

        type: "bar",

        data: {

            labels: data.labels,

            datasets: [{

                label: "Average GPA",

                data: data.values,

                borderWidth: 1

            }]

        },

        options: {

            responsive: true,

            maintainAspectRatio: false,

            scales: {

                y: {

                    beginAtZero: true,

                    max: 4

                }

            }

        }

    });

}


/* =====================================================
   ACTIVITIES VS GPA
   ===================================================== */

function createActivitiesChart(data) {

    const canvas =
        document.getElementById(
            "activitiesChart"
        );

    if (!canvas) return;


    const activities = Object.keys(data);

    const labels = [
        "No",
        "Yes"
    ];


    const datasets = activities.map(
        function (activity) {

            return {

                label: activity,

                data: labels.map(
                    function (label) {

                        const values =
                            data[activity];

                        const index =
                            values.labels.indexOf(
                                label
                            );

                        if (index === -1) {
                            return 0;
                        }

                        return values.values[index];

                    }
                ),

                borderWidth: 1

            };

        }
    );


    new Chart(canvas, {

        type: "bar",

        data: {

            labels: labels,

            datasets: datasets

        },

        options: {

            responsive: true,

            maintainAspectRatio: false,

            scales: {

                y: {

                    beginAtZero: true,

                    max: 4,

                    title: {

                        display: true,

                        text: "Average GPA"

                    }

                }

            }

        }

    });

}