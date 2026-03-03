const SUPPORTED_EXTENSIONS = new Set([".txt", ".md", ".csv", ".json", ".log", ".pdf", ".doc", ".docx"]);
const MAX_TOTAL_UPLOAD_BYTES = 24 * 1024 * 1024;

let isSubmitting = false;
let progressInterval = null;
let canRetry = false;
let currentUserId = "";
const LEGACY_REPORT_KEY = "latest_analysis_report";
const REPORT_KEY_PREFIX = "latest_analysis_report_user_";

function reportStorageKey() {
    return currentUserId ? `${REPORT_KEY_PREFIX}${currentUserId}` : LEGACY_REPORT_KEY;
}

function clearReportCache() {
    localStorage.removeItem(LEGACY_REPORT_KEY);
    Object.keys(localStorage).forEach((key) => {
        if (key.startsWith(REPORT_KEY_PREFIX)) {
            localStorage.removeItem(key);
        }
    });
}

function clearLegacyReportCache() {
    localStorage.removeItem(LEGACY_REPORT_KEY);
}

function setCachedReport(report) {
    if (!report) {
        return;
    }
    localStorage.setItem(reportStorageKey(), JSON.stringify(report));
}

function setStatus(message, type = "") {
    const statusEl = document.getElementById("submitStatus");
    if (!statusEl) {
        return;
    }
    statusEl.className = `status ${type}`.trim();
    statusEl.textContent = message;
}

function selectedMode() {
    const checked = document.querySelector("input[name='analysisMode']:checked");
    return checked ? checked.value : "full";
}

function collectSourceTexts() {
    return Array.from(document.querySelectorAll(".source-input"))
        .map((node) => node.value.trim())
        .filter((text) => text.length > 0);
}

function addSourceTextarea() {
    const list = document.getElementById("sourcesList");
    if (!list) {
        return;
    }
    const textarea = document.createElement("textarea");
    textarea.className = "source-input";
    textarea.placeholder = `Paste source/reference text #${list.children.length + 1}`;
    list.appendChild(textarea);
}

function getFileExtension(name) {
    const idx = (name || "").lastIndexOf(".");
    if (idx < 0) {
        return "";
    }
    return name.slice(idx).toLowerCase();
}

function formatBytes(bytes) {
    const size = Number(bytes || 0);
    if (size < 1024) {
        return `${size} B`;
    }
    if (size < 1024 * 1024) {
        return `${(size / 1024).toFixed(2)} KB`;
    }
    return `${(size / (1024 * 1024)).toFixed(2)} MB`;
}

function renderFileChips(containerId, files) {
    const container = document.getElementById(containerId);
    if (!container) {
        return;
    }
    container.innerHTML = "";
    if (!files || files.length === 0) {
        return;
    }
    const fragment = document.createDocumentFragment();
    files.forEach((file) => {
        const chip = document.createElement("span");
        chip.className = "file-chip";
        chip.textContent = `${file.name} (${formatBytes(file.size)})`;
        fragment.appendChild(chip);
    });
    container.appendChild(fragment);
}

function assignFilesToInput(input, files) {
    if (!input || !files) {
        return;
    }
    try {
        const transfer = new DataTransfer();
        Array.from(files).forEach((file) => transfer.items.add(file));
        input.files = transfer.files;
    } catch (error) {
        // Browser may not allow programmatic assignment; in that case ignore.
    }
}

function setupDropZone(zoneId, inputId, multiple = false) {
    const zone = document.getElementById(zoneId);
    const input = document.getElementById(inputId);
    if (!zone || !input) {
        return;
    }

    zone.addEventListener("click", () => input.click());
    zone.addEventListener("dragover", (event) => {
        event.preventDefault();
        zone.classList.add("drop-zone-active");
    });
    zone.addEventListener("dragleave", () => {
        zone.classList.remove("drop-zone-active");
    });
    zone.addEventListener("drop", (event) => {
        event.preventDefault();
        zone.classList.remove("drop-zone-active");
        const dropped = Array.from(event.dataTransfer?.files || []);
        if (!dropped.length) {
            return;
        }
        const files = multiple ? dropped : [dropped[0]];
        assignFilesToInput(input, files);
        input.dispatchEvent(new Event("change"));
    });
}

function fileValidationErrors(files) {
    const errors = [];
    let totalBytes = 0;
    files.forEach((file) => {
        const ext = getFileExtension(file.name);
        totalBytes += Number(file.size || 0);
        if (!SUPPORTED_EXTENSIONS.has(ext)) {
            errors.push(`Unsupported file type for ${file.name}`);
        }
    });
    if (totalBytes > MAX_TOTAL_UPLOAD_BYTES) {
        errors.push(`Total file size exceeds ${formatBytes(MAX_TOTAL_UPLOAD_BYTES)} limit.`);
    }
    return errors;
}

function toggleReferenceSection() {
    const mode = selectedMode();
    const refSection = document.getElementById("referenceSection");
    if (!refSection) {
        return;
    }
    const hidden = mode === "ai_only";
    refSection.classList.toggle("hidden", hidden);
}

function setSubmittingState(submitting) {
    isSubmitting = submitting;
    const analyzeBtn = document.getElementById("analyzeBtn");
    const retryBtn = document.getElementById("retryBtn");
    if (analyzeBtn) {
        analyzeBtn.disabled = submitting;
    }
    if (retryBtn) {
        retryBtn.disabled = submitting || !canRetry;
    }
}

function updateProgress(value, text = "") {
    const wrap = document.getElementById("uploadProgressWrap");
    const bar = document.getElementById("uploadProgressBar");
    const label = document.getElementById("uploadProgressText");
    if (!wrap || !bar || !label) {
        return;
    }
    wrap.classList.remove("hidden");
    bar.style.width = `${Math.min(100, Math.max(0, value))}%`;
    if (text) {
        label.textContent = text;
    }
}

function hideProgress() {
    const wrap = document.getElementById("uploadProgressWrap");
    const bar = document.getElementById("uploadProgressBar");
    if (!wrap || !bar) {
        return;
    }
    wrap.classList.add("hidden");
    bar.style.width = "0%";
}

function startProgress() {
    updateProgress(8, "Uploading and analyzing...");
    clearInterval(progressInterval);
    let current = 8;
    progressInterval = setInterval(() => {
        current = Math.min(current + 6, 90);
        updateProgress(current);
    }, 450);
}

function stopProgress(success) {
    clearInterval(progressInterval);
    progressInterval = null;
    if (success) {
        updateProgress(100, "Completed.");
        window.setTimeout(() => hideProgress(), 600);
        return;
    }
    updateProgress(100, "Request failed.");
}

function validateSubmission(mode, targetText, targetFile, sourceTexts, sourceFiles) {
    const errors = [];
    if (!targetText && !targetFile) {
        errors.push("Provide target text or upload one target file.");
    }
    if (mode === "full" && sourceTexts.length === 0 && sourceFiles.length === 0) {
        errors.push("Full mode requires at least one reference text or reference file.");
    }

    const targetErrors = targetFile ? fileValidationErrors([targetFile]) : [];
    const sourceErrors = sourceFiles.length ? fileValidationErrors(sourceFiles) : [];
    return errors.concat(targetErrors, sourceErrors);
}

async function runAnalysis() {
    if (isSubmitting) {
        return;
    }

    const mode = selectedMode();
    const targetText = document.getElementById("targetText").value.trim();
    const targetFileInput = document.getElementById("targetFile");
    const sourceFileInput = document.getElementById("sourceFiles");
    const targetFile = targetFileInput?.files?.[0] || null;
    const sourceFiles = Array.from(sourceFileInput?.files || []);
    const sourceTexts = collectSourceTexts();

    const validationErrors = validateSubmission(mode, targetText, targetFile, sourceTexts, sourceFiles);
    if (validationErrors.length > 0) {
        setStatus(validationErrors.join(" "), "error");
        canRetry = true;
        setSubmittingState(false);
        return;
    }

    const form = new FormData();
    form.append("analysis_mode", mode);
    if (targetFile) {
        form.append("document", targetFile);
    } else {
        form.append("text", targetText);
    }
    form.append("source_texts", JSON.stringify(sourceTexts));
    sourceFiles.forEach((file) => form.append("sources", file));

    setSubmittingState(true);
    setStatus("Submitting analysis request...");
    startProgress();

    try {
        const response = await fetch("/api/analyze", { method: "POST", body: form });
        if (response.status === 401) {
            window.location.href = "/login";
            return;
        }
        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.error || payload.detail || "Analysis failed");
        }
        canRetry = false;
        stopProgress(true);
        setCachedReport(payload);
        setStatus("Analysis completed. Redirecting to result page...", "success");
        window.setTimeout(() => {
            window.location.href = "/result";
        }, 450);
    } catch (error) {
        canRetry = true;
        stopProgress(false);
        setStatus(`Analysis failed: ${error.message}`, "error");
    } finally {
        setSubmittingState(false);
    }
}

function bindFileInputs() {
    const targetInput = document.getElementById("targetFile");
    const sourceInput = document.getElementById("sourceFiles");

    targetInput.addEventListener("change", () => {
        renderFileChips("targetFileChips", Array.from(targetInput.files || []));
    });
    sourceInput.addEventListener("change", () => {
        renderFileChips("sourceFileChips", Array.from(sourceInput.files || []));
    });
}

function init() {
    setupDropZone("targetDropZone", "targetFile", false);
    setupDropZone("sourceDropZone", "sourceFiles", true);
    bindFileInputs();
    toggleReferenceSection();

    document.querySelectorAll("input[name='analysisMode']").forEach((node) => {
        node.addEventListener("change", toggleReferenceSection);
    });
    document.getElementById("addSourceBtn").addEventListener("click", addSourceTextarea);
    document.getElementById("analysisForm").addEventListener("submit", (event) => {
        event.preventDefault();
        runAnalysis();
    });
    document.getElementById("retryBtn").addEventListener("click", () => runAnalysis());
    const logoutBtn = document.getElementById("logoutBtn");
    if (logoutBtn) {
        logoutBtn.addEventListener("click", async () => {
            try {
                await fetch("/api/auth/logout", { method: "POST" });
            } finally {
                clearReportCache();
                window.location.href = "/login";
            }
        });
    }
}

async function ensureAuthenticated() {
    try {
        const response = await fetch("/api/auth/me", { method: "GET" });
        if (!response.ok) {
            window.location.href = "/login";
            return false;
        }
        const payload = await response.json();
        currentUserId = String(payload?.user?.id || "");
        clearLegacyReportCache();
        return true;
    } catch (_error) {
        window.location.href = "/login";
        return false;
    }
}

ensureAuthenticated().then((ok) => {
    if (ok) {
        init();
    }
});
