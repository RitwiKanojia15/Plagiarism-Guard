function setStatus(message, type = "") {
    const node = document.getElementById("authStatus");
    if (!node) {
        return;
    }
    node.className = `status ${type}`.trim();
    node.textContent = message;
}

function oauthErrorMessage(code) {
    const mapping = {
        google_not_configured: "Google login is not configured on the server yet.",
        google_state_mismatch: "Google login session expired. Please try again.",
        google_missing_code: "Google did not return an authorization code.",
        google_token_exchange_failed: "Google token exchange failed. Please retry.",
        google_missing_access_token: "Google login failed: missing access token.",
        google_profile_fetch_failed: "Could not fetch your Google profile.",
        google_invalid_email: "Google account email is missing or not verified.",
        google_access_denied: "Google login was canceled.",
        google_callback_failed: "Google login failed unexpectedly. Please retry.",
    };
    return mapping[code] || "Google login failed. Please try again.";
}

function showOauthErrorFromQuery() {
    const params = new URLSearchParams(window.location.search);
    const oauthError = params.get("oauth_error");
    if (!oauthError) {
        return;
    }
    setStatus(oauthErrorMessage(oauthError), "error");
}

async function submitLogin(event) {
    event.preventDefault();
    const email = document.getElementById("email").value.trim();
    const password = document.getElementById("password").value;

    if (!email || !password) {
        setStatus("Email and password are required.", "error");
        return;
    }

    setStatus("Authenticating...");
    try {
        const response = await fetch("/api/auth/login", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email, password }),
        });
        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.error || "Login failed");
        }
        localStorage.removeItem("latest_analysis_report");
        setStatus("Login successful. Redirecting...", "success");
        window.location.href = "/dashboard";
    } catch (error) {
        setStatus(error.message, "error");
    }
}

document.getElementById("loginForm").addEventListener("submit", submitLogin);
showOauthErrorFromQuery();
