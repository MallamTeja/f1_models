# Security Policy

This document explains how to report security issues in **f1_models**, an end-to-end F1 race prediction system built with Python, FastAPI, and machine learning models.

Security matters even for a portfolio or hobby project, especially around API endpoints, environment variables, and third-party dependencies.

---

## Supported Versions

This project does not currently publish versioned releases. Security fixes are applied to the latest code on the `main` branch.

| Target            | Supported        |
|-------------------|------------------|
| `main` branch     | Yes (actively fixed) |
| Other branches    | No               |
| Old commits/tags  | No               |

If a security issue affects a deployed environment (for example, a Render service built from this repo), it will be patched on `main` and redeployed as soon as reasonably possible.

---

## Reporting a Vulnerability

If you discover a security vulnerability, do not open a public GitHub issue or discuss it in public channels.

Instead, please:

1. Email the maintainer at: `your-security-email@example.com`  
   (Replace this with a real email address or alias that you actually monitor.)

2. Use a clear subject line, for example:  
   `f1_models SECURITY: [short description]`

3. Include the following details in your report:
   - A description of the issue and why you believe it is a security vulnerability.
   - Step-by-step instructions to reproduce the issue (requests, payloads, environment details).
   - Which components are affected (for example, a FastAPI endpoint in `main.py`, a CI workflow, or a deployment configuration).
   - The potential impact (such as data exposure, remote code execution, denial of service, or model tampering).
   - Any proof-of-concept exploit or logs that can help verify the issue.

4. If the vulnerability involves secrets (API keys, tokens, environment variables), please:
   - Redact sensitive values in screenshots or logs.
   - Mention exactly where you saw the secret (for example, a specific commit, GitHub Actions log, or deployment panel).

---

## What to Expect After Reporting

- You should receive an acknowledgment within 48 hours.
- A basic assessment and response plan (accept or reject, severity, and next steps) should be shared within 5 to 7 days.
- For accepted issues:
  - A fix will be developed and tested (locally and through CI, if configured).
  - Any deployed services using this repository will be updated.
  - A brief note may be added to the README or release notes describing the fix, without exposing exploit details.
- If you would like recognition, you can be credited in commit messages or documentation. Anonymous reporting is also welcome.

---

## Scope and Typical Risks for This Project

The following areas are considered in scope for security reporting:

- **FastAPI application and endpoints**
  - Injection vulnerabilities (for example, command or SQL) if databases or shell calls are introduced.
  - Insecure request handling or missing validation of user input to prediction endpoints.
  - Missing or weak authentication and authorization if private endpoints are added later.

- **Machine learning and data handling**
  - Tampering with model files (`*.keras`, `*.json`) or lookup data (`lookup_data.json`) to influence predictions.
  - Poisoning of training data pipelines if automated ingestion is introduced.
  - Exposure of any sensitive training data, if such data is ever added.

- **Configuration and deployment**
  - Leaked secrets or environment variables (for example, `.env`, CI secrets, or Render configuration).
  - Insecure CI/CD workflows (GitHub Actions) that allow arbitrary code execution from untrusted pull requests.
  - Misconfigured CORS, HTTP, or TLS settings in production deployments.

- **Dependencies and supply chain**
  - Known vulnerabilities in dependencies listed in `requirements.txt` or `req.txt`.
  - Insecure or unpinned dependencies that could be compromised upstream.

Out of scope (for now):

- Purely theoretical attacks that require unrealistic access (for example, root access on the deployment host).
- Issues that only affect forks or heavily modified versions of this project.

---

## Recommendations for Contributors

If you contribute to **f1_models**, please help keep it secure by:

- Avoiding hard-coded secrets or tokens in code or commits.
- Using a `.env` file for sensitive values and never committing your real `.env`.
- Running dependency checks (for example, with `pip-audit` or similar tools) before major changes.
- Being careful when modifying CI workflows or deployment files (such as `.github/workflows` or `render.yaml`).

---

Thank you for helping keep **f1_models** and its users safe.
