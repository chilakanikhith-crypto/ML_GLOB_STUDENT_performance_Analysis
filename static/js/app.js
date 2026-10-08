document.addEventListener("DOMContentLoaded", function () {

    // Highlight the current sidebar page
    const currentPath = window.location.pathname;

    document.querySelectorAll(".nav-item").forEach(function (link) {

        const linkPath = link.getAttribute("href");

        if (
            linkPath === currentPath ||
            (currentPath === "/" && linkPath === "/")
        ) {
            link.classList.add("active");
        }
    });


    // Automatically hide Bootstrap alerts after 5 seconds
    const alerts = document.querySelectorAll(".alert");

    alerts.forEach(function (alert) {

        setTimeout(function () {

            const closeButton =
                alert.querySelector(".btn-close");

            if (closeButton) {
                closeButton.click();
            }

        }, 5000);
    });

});