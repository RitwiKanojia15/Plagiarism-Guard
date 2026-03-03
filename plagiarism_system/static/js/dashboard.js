function setStatus(message, type = "") {
    const statusEl = document.getElementById("healthStatus");
    if (!statusEl) {
        return;
    }
    statusEl.className = `status ${type}`.trim();
    statusEl.textContent = message;
}

function clearReportCache() {
    localStorage.removeItem("latest_analysis_report");
    Object.keys(localStorage).forEach((key) => {
        if (key.startsWith("latest_analysis_report_user_")) {
            localStorage.removeItem(key);
        }
    });
}

function userInitials(fullName) {
    const parts = String(fullName || "")
        .trim()
        .split(/\s+/)
        .filter(Boolean);
    if (!parts.length) {
        return "U";
    }
    if (parts.length === 1) {
        return parts[0].slice(0, 2).toUpperCase();
    }
    return `${parts[0][0]}${parts[1][0]}`.toUpperCase();
}

function renderWelcome(user) {
    const title = document.getElementById("welcomeTitle");
    const subtitle = document.getElementById("welcomeSubtext");
    if (title) {
        title.textContent = `Welcome, ${user.full_name}`;
    }
    if (subtitle) {
        subtitle.textContent = `Signed in as ${user.email}. Upload PDF, DOC, DOCX or plain text for analysis.`;
    }
}

function renderProfile(user) {
    const initials = userInitials(user.full_name);

    const triggerName = document.getElementById("profileTriggerName");
    const avatar = document.getElementById("profileAvatar");
    const avatarLarge = document.getElementById("profileAvatarLarge");
    const fullName = document.getElementById("profileFullName");
    const email = document.getElementById("profileEmail");

    if (triggerName) {
        triggerName.textContent = user.full_name || "Profile";
    }
    if (avatar) {
        avatar.textContent = initials;
    }
    if (avatarLarge) {
        avatarLarge.textContent = initials;
    }
    if (fullName) {
        fullName.textContent = user.full_name || "User";
    }
    if (email) {
        email.textContent = user.email || "";
    }
}

function renderReports(reports) {
    const host = document.getElementById("recentReports");
    if (!host) {
        return;
    }
    host.innerHTML = "";
    if (!reports || reports.length === 0) {
        host.innerHTML = "<p class='small'>No reports yet. Run your first analysis from Upload page.</p>";
        return;
    }

    const table = document.createElement("table");
    table.className = "report-table";
    table.innerHTML = `
        <thead>
            <tr>
                <th>File</th>
                <th>Mode</th>
                <th>Total Similarity</th>
                <th>AI Likelihood</th>
                <th>Created</th>
                <th>PDF</th>
            </tr>
        </thead>
        <tbody></tbody>
    `;
    const tbody = table.querySelector("tbody");
    reports.forEach((row) => {
        const tr = document.createElement("tr");
        const pdfCell = row.pdf_download_url
            ? `<a class="auth-link" href="${row.pdf_download_url}" target="_blank" rel="noopener">Download</a>`
            : "-";
        tr.innerHTML = `
            <td>${row.file_name || "-"}</td>
            <td>${String(row.analysis_mode || "").toUpperCase()}</td>
            <td>${Number(row.total_similarity || 0).toFixed(2)}%</td>
            <td>${Number(row.ai_likelihood || 0).toFixed(2)}%</td>
            <td>${row.created_at ? new Date(row.created_at).toLocaleString() : "-"}</td>
            <td>${pdfCell}</td>
        `;
        tbody.appendChild(tr);
    });
    host.appendChild(table);
}

async function loadDashboard() {
    setStatus("Checking authentication and loading dashboard...");
    try {
        const meRes = await fetch("/api/auth/me", { method: "GET" });
        const meData = await meRes.json();
        if (!meRes.ok) {
            window.location.href = "/login";
            return;
        }
        renderWelcome(meData.user);
        renderProfile(meData.user);

        const [healthRes, reportsRes] = await Promise.all([
            fetch("/api/health", { method: "GET" }),
            fetch("/api/reports", { method: "GET" }),
        ]);

        const healthData = await healthRes.json();
        if (!healthRes.ok) {
            throw new Error(healthData.error || "Health check failed");
        }
        setStatus(`Service online: ${healthData.service}`, "success");

        const reportsData = await reportsRes.json();
        if (reportsRes.ok) {
            renderReports(reportsData.reports || []);
        }
    } catch (error) {
        setStatus(`Dashboard load failed: ${error.message}`, "error");
    }
}

async function logout() {
    try {
        await fetch("/api/auth/logout", { method: "POST" });
    } finally {
        clearReportCache();
        window.location.href = "/login";
    }
}

function bindLogoutButtons() {
    document.querySelectorAll("[data-logout-btn]").forEach((button) => {
        button.addEventListener("click", logout);
    });
}

function bindProfileMenu() {
    const menu = document.getElementById("profileMenu");
    const trigger = document.getElementById("profileMenuBtn");
    if (!menu || !trigger) {
        return;
    }

    const closeMenu = () => {
        menu.classList.remove("profile-menu-open");
        trigger.setAttribute("aria-expanded", "false");
    };

    trigger.addEventListener("click", (event) => {
        event.stopPropagation();
        const open = menu.classList.toggle("profile-menu-open");
        trigger.setAttribute("aria-expanded", String(open));
    });

    document.addEventListener("click", (event) => {
        if (!menu.contains(event.target)) {
            closeMenu();
        }
    });

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
            closeMenu();
        }
    });
}

bindLogoutButtons();
bindProfileMenu();
loadDashboard();
