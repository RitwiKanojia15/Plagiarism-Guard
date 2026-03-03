function setStatus(message, type = "") {
    const node = document.getElementById("authStatus");
    if (!node) {
        return;
    }
    node.className = `status ${type}`.trim();
    node.textContent = message;
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

