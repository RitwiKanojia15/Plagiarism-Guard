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
        google_not_configured: "Google signup is not configured on the server yet.",
        google_state_mismatch: "Google session expired. Please try again.",
        google_missing_code: "Google did not return an authorization code.",
        google_token_exchange_failed: "Google token exchange failed. Please retry.",
        google_missing_access_token: "Google signup failed: missing access token.",
        google_profile_fetch_failed: "Could not fetch your Google profile.",
        google_invalid_email: "Google account email is missing or not verified.",
        google_access_denied: "Google signup was canceled.",
        google_callback_failed: "Google signup failed unexpectedly. Please retry.",
    };
    return mapping[code] || "Google signup failed. Please try again.";
}

function showOauthErrorFromQuery() {
    const params = new URLSearchParams(window.location.search);
    const oauthError = params.get("oauth_error");
    if (!oauthError) {
        return;
    }
    setStatus(oauthErrorMessage(oauthError), "error");
}

async function submitRegistration(event) {
    event.preventDefault();
    const fullName = document.getElementById("fullName").value.trim();
    const email = document.getElementById("email").value.trim();
    const password = document.getElementById("password").value;

    if (!fullName || !email || !password) {
        setStatus("Full name, email and password are required.", "error");
        return;
    }
    if (password.length < 8) {
        setStatus("Password must be at least 8 characters.", "error");
        return;
    }

    setStatus("Creating account...");
    try {
        const response = await fetch("/api/auth/register", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ full_name: fullName, email, password }),
        });
        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.error || "Registration failed");
        }
        setStatus("Registration successful. Redirecting to login...", "success");
        setTimeout(() => {
            window.location.href = "/login";
        }, 500);
    } catch (error) {
        setStatus(error.message, "error");
    }
}

document.getElementById("registerForm").addEventListener("submit", submitRegistration);
showOauthErrorFromQuery();
