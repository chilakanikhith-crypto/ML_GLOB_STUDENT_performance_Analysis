document.addEventListener("DOMContentLoaded", function () {

    // Highlight the current sidebar page.
    const currentPath = window.location.pathname;

    document.querySelectorAll(".nav-item").forEach(function (link) {
        const linkPath = link.getAttribute("href");

        if (!linkPath) {
            return;
        }

        // Match the current route and remove trailing slashes.
        const normalizedCurrentPath =
            currentPath.length > 1
                ? currentPath.replace(/\/+$/, "")
                : currentPath;

        const normalizedLinkPath =
            linkPath.length > 1
                ? linkPath.replace(/\/+$/, "")
                : linkPath;

        if (normalizedLinkPath === normalizedCurrentPath) {
            link.classList.add("active");
            link.setAttribute("aria-current", "page");
        } else {
            link.classList.remove("active");
            link.removeAttribute("aria-current");
        }
    });

    // Automatically hide Bootstrap alerts after 5 seconds.
    const alerts = document.querySelectorAll(".alert");

    alerts.forEach(function (alert) {
        // Avoid auto-closing persistent alerts.
        if (alert.dataset.autoDismiss === "false") {
            return;
        }

        setTimeout(function () {
            // Do nothing if the alert has already been removed.
            if (!alert.isConnected) {
                return;
            }

            // Use Bootstrap's alert component when available.
            if (
                window.bootstrap &&
                window.bootstrap.Alert
            ) {
                const bootstrapAlert =
                    window.bootstrap.Alert.getOrCreateInstance(alert);

                bootstrapAlert.close();
                return;
            }

            // Fallback for alerts without Bootstrap JavaScript.
            alert.style.transition = "opacity 0.3s ease";
            alert.style.opacity = "0";

            setTimeout(function () {
                alert.remove();
            }, 300);

        }, 5000);
    });

});