function setStatus(message, type = "") {
    const node = document.getElementById("authStatus");
    if (!node) {
        return;
    }
    node.className = `status ${type}`.trim();
    node.textContent = message;
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
