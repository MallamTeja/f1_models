# Contributing to f1_models

Thank you for your interest in improving **f1_models**, an end-to-end F1 race winner prediction system built with Python, FastAPI, and modern machine learning tooling.  
This document explains how to propose changes, report bugs, and add new features in a consistent way.

---

## Code of conduct

By participating in this project, you agree to uphold a respectful, constructive, and inclusive environment.  
Be kind, stay on topic (F1 and data), and assume good intent.

If you experience or witness unacceptable behavior, please open a confidential issue or contact the maintainer directly via the email address listed on the maintainer’s GitHub profile.

---

## How to ask questions

Use **GitHub Issues** for:

- Bug reports.
- Feature requests.
- Clarifications about implementation details (API, models, data schema).

Before opening a new issue, please:

- Search existing issues to avoid duplicates.
- Include enough context: logs, error messages, operating system, Python version, and clear steps to reproduce the problem.

---

## Getting started (local setup)

1. **Fork** the repository and clone your fork.
2. Create and activate a virtual environment:
   - `python -m venv .venv && source .venv/bin/activate` (Linux/macOS)
   - `python -m venv .venv && .venv\\Scripts\\activate` (Windows)
3. Install dependencies:
   - `pip install -r requirements.txt` or `pip install -r req.txt` (depending on which file is used in the repo).
4. Copy `.env.example` to `.env` and fill in the required values (API keys, secrets, and other configuration).
5. Run the API locally:
   - `uvicorn main:app --reload`
6. Run tests to ensure everything passes before you make changes:
   - `pytest` or `python -m pytest` (if tests are configured).

---

## How to contribute

### 1. Reporting bugs

When reporting a bug, include:

- A clear, descriptive title.
- Steps to reproduce (ideally, a minimal example).
- Expected versus actual behavior.
- Logs, error messages, and screenshots if relevant.
- Environment details: operating system, Python version, and dependency versions if known.

Use the `bug` label if it is available.

### 2. Suggesting features or improvements

For new ideas (models, features, visualizations, MLOps improvements), open an issue labeled `enhancement` and include:

- The problem or use case.
- Your proposed solution or approach.
- Any alternatives you considered.
- Potential impact on performance, complexity, or user experience.

This helps discuss and align on the design before you invest time in code.

### 3. Working on issues

Once an issue is approved or assigned:

1. Create a feature branch from `main`:
   - `git checkout -b feature/my-feature-name`
2. Make changes in small, focused commits with meaningful messages.
3. Keep your branch up to date with `main`:
   - `git fetch origin && git rebase origin/main`
4. Ensure:
   - All tests pass.
   - Linting and formatting are clean (see “Code style and standards” below).
5. Push your branch and open a **Pull Request (PR)** against `main`.

In your PR description:

- Reference related issues (for example, `Closes #12`).
- Describe what changed and why.
- Add screenshots, plots, or metrics for visual changes or modeling results if helpful.

---

## Code style and standards

To keep the codebase consistent and easy to read:

- Follow PEP 8 for Python code style.
- Use type hints wherever possible for new functions and modules.
- Prefer clear, descriptive names over abbreviations (for example, `race_results_df` instead of `rrdf`).
- Keep functions focused; avoid large “god scripts”. Split logic into smaller utilities when needed.

If this project adopts linters or formatters (for example, `black`, `isort`, or `ruff`), run them before committing.

---

## Tests and quality checks

High-quality predictions and stable APIs are critical for this project.

- Add or update tests when you:
  - Add new APIs (FastAPI endpoints).
  - Change preprocessing or feature engineering logic.
  - Introduce or replace models.
- Use existing test files (such as `test_main.py` and `inferencetest.py`) as references where applicable.

A pull request may be blocked or rejected until:

- Tests pass locally.
- Continuous integration (CI) checks (for example, GitHub Actions) pass successfully.

---

## Contributions specific to this project

### Model and machine learning contributions

If you are contributing to the machine learning side (new models, features, or training scripts):

- Document:
  - Data sources and assumptions.
  - Target variable(s) and training/validation splits.
  - Evaluation metrics (for example, accuracy, log-loss, Brier score).
- Include a short model card in the PR description or in a separate markdown file, covering:
  - Strengths and limitations.
  - Tracks or race conditions where the model performs poorly.
  - Any known biases or caveats.

### API and MLOps contributions

For API, infrastructure, or deployment work:

- Keep endpoints backward compatible whenever possible.
- Document breaking changes clearly in the PR.
- Update or add documentation for:
  - New endpoints (path, method, request/response schema).
  - Environment variables.
  - Deployment configuration (for example, `render.yaml` and CI workflows).

---

## Documentation improvements

Good documentation is a valuable contribution.

- Fix typos, clarify explanations, or add examples to the README and other docs.
- Ensure code snippets are runnable and consistent with the current API and project structure.
- You can open small PRs just for documentation improvements; these are welcome.

---

## License

By contributing to **f1_models**, you agree that your contributions will be licensed under the same license as this repository (see the `LICENSE` file for details).

---

## Acknowledgements

Thank you for helping make **f1_models** a better F1 race prediction platform for fans, analysts, and machine learning practitioners.
