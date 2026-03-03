const THEME_STORAGE_KEY = "pg_theme_mode";
const DARK_THEME = "dark";
const LIGHT_THEME = "light";

function isValidTheme(value) {
    return value === DARK_THEME || value === LIGHT_THEME;
}

function storedTheme() {
    const value = localStorage.getItem(THEME_STORAGE_KEY);
    return isValidTheme(value) ? value : "";
}

function systemPreferredTheme() {
    return window.matchMedia("(prefers-color-scheme: light)").matches ? LIGHT_THEME : DARK_THEME;
}

function activeTheme() {
    return storedTheme() || systemPreferredTheme();
}

function syncToggleLabels(theme) {
    const toggleButtons = document.querySelectorAll("[data-theme-toggle]");
    const nextLabel = theme === DARK_THEME ? "Switch to Light" : "Switch to Dark";
    toggleButtons.forEach((button) => {
        button.textContent = nextLabel;
        button.setAttribute("aria-label", nextLabel);
        button.setAttribute("aria-pressed", String(theme === LIGHT_THEME));
    });
}

function applyTheme(theme) {
    const normalized = isValidTheme(theme) ? theme : DARK_THEME;
    document.documentElement.setAttribute("data-theme", normalized);
    syncToggleLabels(normalized);
    window.dispatchEvent(new CustomEvent("themechange", { detail: { theme: normalized } }));
}

function toggleTheme() {
    const current = document.documentElement.getAttribute("data-theme") || activeTheme();
    const nextTheme = current === DARK_THEME ? LIGHT_THEME : DARK_THEME;
    localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
    applyTheme(nextTheme);
}

function initTheme() {
    applyTheme(activeTheme());
    const toggleButtons = document.querySelectorAll("[data-theme-toggle]");
    toggleButtons.forEach((button) => {
        button.addEventListener("click", toggleTheme);
    });
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initTheme);
} else {
    initTheme();
}
