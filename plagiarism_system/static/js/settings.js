function setStatus(message, type = "") {
    const statusEl = document.getElementById("settingsStatus");
    if (!statusEl) {
        return;
    }
    statusEl.className = `status ${type}`.trim();
    statusEl.textContent = message;
}

function updateUser(user) {
    const nameEl = document.getElementById("settingsName");
    const emailEl = document.getElementById("settingsEmail");
    const createdEl = document.getElementById("settingsCreated");

    if (nameEl) {
        nameEl.textContent = user.full_name || "-";
    }
    if (emailEl) {
        emailEl.textContent = user.email || "-";
    }
    if (createdEl) {
        createdEl.textContent = user.created_at ? new Date(user.created_at).toLocaleString() : "-";
    }
}

async function loadProfile() {
    setStatus("Loading account settings...");
    try {
        const response = await fetch("/api/auth/me", { method: "GET" });
        if (response.status === 401) {
            window.location.href = "/login";
            return;
        }

        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.error || "Failed to load profile.");
        }

        updateUser(payload.user || {});
        setStatus("Account loaded.", "success");
    } catch (error) {
        setStatus(`Settings load failed: ${error.message}`, "error");
    }
}

async function logout() {
    try {
        await fetch("/api/auth/logout", { method: "POST" });
    } finally {
        window.location.href = "/login";
    }
}

function bindLogoutButtons() {
    const primaryLogout = document.getElementById("logoutBtn");
    if (primaryLogout) {
        primaryLogout.addEventListener("click", logout);
    }

    const secondaryLogout = document.getElementById("secondaryLogoutBtn");
    if (secondaryLogout) {
        secondaryLogout.addEventListener("click", logout);
    }
}

bindLogoutButtons();
loadProfile();
