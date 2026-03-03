const chartState = {
    aiProbabilityChart: null,
    stylometryRadarChart: null,
    featureImportanceChart: null,
    similarityHeatmap: null,
};

function isLightTheme() {
    return document.documentElement.getAttribute("data-theme") === "light";
}

function applyChartDefaults() {
    if (!window.Chart) {
        return;
    }
    const light = isLightTheme();
    Chart.defaults.color = light ? "#334155" : "#cfe0fb";
    Chart.defaults.borderColor = light ? "rgba(123, 151, 192, 0.45)" : "rgba(114, 146, 197, 0.35)";
    Chart.defaults.font.family = "'Share Tech Mono', 'JetBrains Mono', monospace";
}
applyChartDefaults();

let currentFilter = "all";
let currentReport = null;
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

function getCachedReport() {
    const raw = localStorage.getItem(reportStorageKey());
    if (!raw) {
        return null;
    }
    try {
        return JSON.parse(raw);
    } catch (error) {
        return null;
    }
}

function setCachedReport(report) {
    if (!report) {
        return;
    }
    localStorage.setItem(reportStorageKey(), JSON.stringify(report));
}

async function fetchAuthenticatedUser() {
    const response = await fetch("/api/auth/me", { method: "GET" });
    if (response.status === 401) {
        window.location.href = "/login";
        return null;
    }
    if (!response.ok) {
        return null;
    }
    const payload = await response.json();
    return payload.user || null;
}

async function fetchLatestReportFromServer() {
    const response = await fetch("/api/reports/latest", { method: "GET" });
    if (response.status === 401) {
        window.location.href = "/login";
        return null;
    }
    if (!response.ok) {
        return null;
    }
    const payload = await response.json();
    return payload.report || null;
}

function setMetric(id, value, suffix = "%") {
    const node = document.getElementById(id);
    if (!node) {
        return;
    }
    const numeric = Number(value || 0);
    node.textContent = `${numeric.toFixed(2)}${suffix}`;
}

function setConfidenceBadge(report) {
    const scoreNode = document.getElementById("confidenceScore");
    const badgeNode = document.getElementById("confidenceBadge");
    if (!scoreNode || !badgeNode) {
        return;
    }
    const confidencePercent = Number(report.confidence || 0) * 100;
    scoreNode.textContent = `${confidencePercent.toFixed(2)}%`;
    badgeNode.className = "badge";
    if (confidencePercent >= 75) {
        badgeNode.classList.add("badge-high");
        badgeNode.textContent = "high confidence";
    } else if (confidencePercent >= 45) {
        badgeNode.classList.add("badge-medium");
        badgeNode.textContent = "medium confidence";
    } else {
        badgeNode.classList.add("badge-low");
        badgeNode.textContent = "low confidence";
    }
}

function setMode(report) {
    const node = document.getElementById("analysisMode");
    if (!node) {
        return;
    }
    const mode = String(report.analysis_mode || "full");
    node.textContent = mode === "ai_only" ? "AI-ONLY" : "FULL";
}

function setDownloadButton(report) {
    const button = document.getElementById("downloadPdfBtn");
    if (!button) {
        return;
    }
    const url = String(report.pdf_download_url || "").trim();
    if (url) {
        button.href = url;
        button.classList.remove("hidden");
    } else {
        button.classList.add("hidden");
        button.removeAttribute("href");
    }
}

function filteredHighlights(highlights, filter) {
    if (filter === "all") {
        return highlights;
    }
    return highlights.filter((item) => item.label === filter);
}

function renderHighlights(highlights, filter) {
    const container = document.getElementById("highlightedSentences");
    if (!container) {
        return;
    }
    container.innerHTML = "";
    const rows = filteredHighlights(highlights || [], filter);
    if (!rows.length) {
        container.textContent = "No sentences found for this filter.";
        return;
    }

    const fragment = document.createDocumentFragment();
    rows.forEach((item) => {
        const span = document.createElement("span");
        span.className = "highlight";
        if (item.label === "copied") {
            span.classList.add("highlight-copied");
        } else if (item.label === "paraphrased") {
            span.classList.add("highlight-paraphrased");
        } else if (item.label === "ai_likely") {
            span.classList.add("highlight-ai");
        } else {
            span.classList.add("highlight-unique");
        }
        span.title = item.reason || "";
        span.textContent = item.sentence || "";
        fragment.appendChild(span);
    });
    container.appendChild(fragment);
}

function resetChart(key) {
    const chart = chartState[key];
    if (chart) {
        chart.destroy();
        chartState[key] = null;
    }
}

function renderAiProbabilityChart(report) {
    const context = document.getElementById("aiProbabilityChart");
    if (!context) {
        return;
    }
    resetChart("aiProbabilityChart");
    const aiProbability = Number(report.ai_detection?.ai_probability || 0) * 100;
    const humanProbability = Number(report.ai_detection?.human_probability || 1) * 100;
    const palette = isLightTheme() ? ["#2563eb", "#0f766e"] : ["#60a5fa", "#34d399"];
    chartState.aiProbabilityChart = new Chart(context, {
        type: "bar",
        data: {
            labels: ["AI", "Human"],
            datasets: [
                {
                    label: "Probability (%)",
                    data: [aiProbability, humanProbability],
                    backgroundColor: palette,
                    borderRadius: 8,
                },
            ],
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: { y: { beginAtZero: true, max: 100 } },
        },
    });
}

function normalizeFeature(value, maxValue) {
    if (maxValue <= 0) {
        return 0;
    }
    return Math.min(1, Number(value || 0) / maxValue);
}

function renderStylometryRadar(report) {
    const context = document.getElementById("stylometryRadarChart");
    if (!context) {
        return;
    }
    resetChart("stylometryRadarChart");
    const features = report.stylometry_features || {};
    const labels = [
        "Type Token Ratio",
        "Hapax Ratio",
        "Avg Sentence Length",
        "Sentence Variance",
        "Function Word Ratio",
        "Passive Voice Ratio",
    ];
    const values = [
        Number(features.type_token_ratio || 0),
        Number(features.hapax_legomena_ratio || 0),
        normalizeFeature(Number(features.avg_sentence_length || 0), 40),
        normalizeFeature(Number(features.sentence_length_variance || 0), 100),
        Number(features.function_word_ratio || 0),
        Number(features.passive_voice_ratio || 0),
    ];
    const radarBorder = isLightTheme() ? "#1d4ed8" : "#60a5fa";
    const radarBackground = isLightTheme() ? "rgba(29, 78, 216, 0.14)" : "rgba(96, 165, 250, 0.18)";
    chartState.stylometryRadarChart = new Chart(context, {
        type: "radar",
        data: {
            labels,
            datasets: [
                {
                    label: "Stylometry Profile",
                    data: values,
                    backgroundColor: radarBackground,
                    borderColor: radarBorder,
                    pointBackgroundColor: radarBorder,
                },
            ],
        },
        options: {
            responsive: true,
            scales: { r: { beginAtZero: true, max: 1 } },
        },
    });
}

function renderFeatureImportance(report) {
    const context = document.getElementById("featureImportanceChart");
    if (!context) {
        return;
    }
    resetChart("featureImportanceChart");

    const rows = (report.ai_explanation?.top_contributing_features || []).slice(0, 10);
    const labels = rows.map((row) => row.feature);
    const values = rows.map((row) => Number(row.impact || row.abs_impact || 0));
    const colors = values.map((value) => {
        if (value >= 0) {
            return isLightTheme() ? "#2563eb" : "#38bdf8";
        }
        return isLightTheme() ? "#d62828" : "#fb7185";
    });
    chartState.featureImportanceChart = new Chart(context, {
        type: "bar",
        data: {
            labels,
            datasets: [
                {
                    label: "Feature Impact",
                    data: values,
                    backgroundColor: colors,
                },
            ],
        },
        options: {
            responsive: true,
            indexAxis: "y",
            plugins: { legend: { display: false } },
        },
    });
}

function renderSimilarityHeatmap(report) {
    const context = document.getElementById("similarityHeatmap");
    const note = document.getElementById("heatmapNote");
    if (!context || !note) {
        return;
    }
    resetChart("similarityHeatmap");

    const matrix = report.semantic_details?.similarity_matrix || [];
    if (!matrix.length || !matrix[0]?.length) {
        note.textContent = report.analysis_mode === "ai_only"
            ? "Heatmap hidden in AI-only mode because no reference corpus was used."
            : "No semantic matrix available.";
        return;
    }
    note.textContent = "";

    const maxRows = Math.min(matrix.length, 30);
    const maxCols = Math.min(matrix[0].length, 30);
    const matrixData = [];
    for (let rowIndex = 0; rowIndex < maxRows; rowIndex += 1) {
        for (let colIndex = 0; colIndex < maxCols; colIndex += 1) {
            matrixData.push({
                x: colIndex,
                y: rowIndex,
                v: Number(matrix[rowIndex][colIndex] || 0),
            });
        }
    }

    chartState.similarityHeatmap = new Chart(context, {
        type: "matrix",
        data: {
            datasets: [
                {
                    data: matrixData,
                    width: ({ chart }) => (chart.chartArea || {}).width / maxCols - 1,
                    height: ({ chart }) => (chart.chartArea || {}).height / maxRows - 1,
                    backgroundColor: (ctx) => {
                        const value = Number(ctx.dataset.data[ctx.dataIndex]?.v || 0);
                        const red = Math.floor(255 * value);
                        const green = Math.floor(255 * (1 - value));
                        return `rgba(${red}, ${green}, 120, 0.9)`;
                    },
                },
            ],
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    type: "linear",
                    min: -0.5,
                    max: maxCols - 0.5,
                    ticks: { stepSize: 1 },
                    title: { display: true, text: "Source Sentence Index" },
                },
                y: {
                    type: "linear",
                    reverse: true,
                    min: -0.5,
                    max: maxRows - 0.5,
                    ticks: { stepSize: 1 },
                    title: { display: true, text: "Target Sentence Index" },
                },
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label(context) {
                            const { x, y, v } = context.raw;
                            return `Target ${y}, Source ${x}: ${Number(v).toFixed(3)}`;
                        },
                    },
                },
            },
        },
    });
}

function activateFilter(filter) {
    currentFilter = filter;
    document.querySelectorAll(".filter-btn").forEach((button) => {
        button.classList.toggle("active", button.dataset.filter === filter);
    });
    renderHighlights(currentReport?.sentence_highlighting || [], currentFilter);
}

function bindFilters() {
    document.querySelectorAll(".filter-btn").forEach((button) => {
        button.addEventListener("click", () => activateFilter(button.dataset.filter));
    });
}

function bindLogout() {
    const logoutBtn = document.getElementById("logoutBtn");
    if (!logoutBtn) {
        return;
    }
    logoutBtn.addEventListener("click", async () => {
        try {
            await fetch("/api/auth/logout", { method: "POST" });
        } finally {
            clearReportCache();
            window.location.href = "/login";
        }
    });
}

async function renderResult() {
    const status = document.getElementById("resultStatus");
    const user = await fetchAuthenticatedUser();
    if (!user) {
        status.className = "status error";
        status.textContent = "Authentication required.";
        return;
    }
    currentUserId = String(user.id || "");
    clearLegacyReportCache();

    currentReport = await fetchLatestReportFromServer();
    if (currentReport) {
        setCachedReport(currentReport);
    } else {
        currentReport = getCachedReport();
    }

    if (!currentReport) {
        status.className = "status error";
        status.textContent = "No report found. Run a new analysis from the Upload page.";
        return;
    }

    setMetric("totalSimilarity", currentReport.total_similarity);
    setMetric("exactPlagiarism", currentReport.exact_plagiarism);
    setMetric("paraphrasedContent", currentReport.paraphrased_content);
    setMetric("aiLikelihood", currentReport.ai_likelihood);
    setConfidenceBadge(currentReport);
    setMode(currentReport);
    setDownloadButton(currentReport);

    renderHighlights(currentReport.sentence_highlighting || [], currentFilter);
    renderAiProbabilityChart(currentReport);
    renderStylometryRadar(currentReport);
    renderSimilarityHeatmap(currentReport);
    renderFeatureImportance(currentReport);

    status.className = "status success";
    status.textContent = `Loaded ${currentReport.analysis_mode === "ai_only" ? "AI-only" : "full"} report.`;
}

bindFilters();
bindLogout();
renderResult();

window.addEventListener("themechange", () => {
    applyChartDefaults();
    if (!currentReport) {
        return;
    }
    renderAiProbabilityChart(currentReport);
    renderStylometryRadar(currentReport);
    renderSimilarityHeatmap(currentReport);
    renderFeatureImportance(currentReport);
});
