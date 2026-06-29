function setStatus(message, type = "") {
    const statusEl = document.getElementById("historyStatus");
    if (!statusEl) {
        return;
    }
    statusEl.className = `status ${type}`.trim();
    statusEl.textContent = message;
}

function renderReports(reports) {
    const host = document.getElementById("historyReports");
    if (!host) {
        return;
    }
    host.innerHTML = "";

    if (!reports || reports.length === 0) {
        const empty = document.createElement("div");
        empty.className = "history-empty";
        empty.textContent = "No history available yet. Run your first scan from New Scan.";
        host.appendChild(empty);
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
        const mode = String(row.analysis_mode || "").toUpperCase();
        const created = row.created_at ? new Date(row.created_at).toLocaleString() : "-";
        const pdfCell = row.pdf_download_url
            ? `<a class="auth-link" href="${row.pdf_download_url}" target="_blank" rel="noopener">Download</a>`
            : "-";

        tr.innerHTML = `
            <td>${row.file_name || "-"}</td>
            <td>${mode}</td>
            <td>${Number(row.total_similarity || 0).toFixed(2)}%</td>
            <td>${Number(row.ai_likelihood || 0).toFixed(2)}%</td>
            <td>${created}</td>
            <td>${pdfCell}</td>
        `;
        tbody.appendChild(tr);
    });

    host.appendChild(table);
}

async function loadHistory() {
    setStatus("Loading history...");
    try {
        const response = await fetch("/api/reports", { method: "GET" });
        if (response.status === 401) {
            window.location.href = "/login";
            return;
        }

        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.error || "Failed to load history.");
        }

        renderReports(payload.reports || []);
        setStatus("History loaded.", "success");
    } catch (error) {
        setStatus(`History load failed: ${error.message}`, "error");
    }
}

async function logout() {
    try {
        await fetch("/api/auth/logout", { method: "POST" });
    } finally {
        window.location.href = "/login";
    }
}

function bindLogout() {
    const logoutBtn = document.getElementById("logoutBtn");
    if (!logoutBtn) {
        return;
    }
    logoutBtn.addEventListener("click", logout);
}

bindLogout();
loadHistory();
