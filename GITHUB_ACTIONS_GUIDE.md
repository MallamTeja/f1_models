# GitHub Actions Complete Guide for ML Projects

## Table of Contents
1. [What are GitHub Actions](#what-are-github-actions)
2. [Core Concepts](#core-concepts)
3. [Anatomy of a Workflow](#anatomy-of-a-workflow)
4. [Production-Level Implementation](#production-level-implementation)
5. [Best Practices for ML Projects](#best-practices-for-ml-projects)

---

## What are GitHub Actions?

GitHub Actions is a CI/CD (Continuous Integration/Continuous Deployment) platform built directly into GitHub. It allows you to automate your software development workflow by running tasks (jobs) in response to events.

### Key Benefits:
- **Automation**: Run tests, linting, builds automatically on every push/PR
- **Quality Control**: Catch bugs before merging to main branch
- **Deployment**: Automatically deploy to production after tests pass
- **Cost-Effective**: Free tier includes 2000 minutes/month for public repos
- **Integrated**: Native integration with GitHub, no external setup needed

---

## Core Concepts

### 1. **Workflows**
- YAML files in `.github/workflows/` directory
- Define the entire CI/CD pipeline
- Triggered by events (push, pull_request, schedule, etc.)

### 2. **Events**
Triggers that start a workflow:
- `push`: Code pushed to repository
- `pull_request`: PR opened or updated
- `schedule`: Cron-based scheduling
- `workflow_dispatch`: Manual trigger from GitHub UI
- `release`: New release published

### 3. **Jobs**
- Independent tasks that run in parallel (by default)
- Can have dependencies (run sequentially)
- Each job runs in a fresh runner instance

### 4. **Steps**
- Individual commands or actions within a job
- Run sequentially within same job
- Can use official actions or custom scripts

### 5. **Runners**
- Virtual machines that execute jobs
- GitHub-hosted: Ubuntu, Windows, macOS
- Self-hosted: Your own servers

### 6. **Actions**
- Reusable units of code (like packages)
- Can be official, community, or custom
- Example: `actions/checkout@v4`, `actions/setup-python@v5`

### 7. **Artifacts**
- Files created during workflow execution
- Can be downloaded or passed between jobs
- Example: Test reports, coverage reports, built binaries

### 8. **Caching**
- Caches dependencies between runs
- Speeds up workflow execution
- Example: Python pip packages, npm modules

---

## Anatomy of a Workflow

```yaml
name: CI                          # Workflow name shown in GitHub UI

on:                              # Events that trigger this workflow
  push:
    branches: [main, develop]     # Only trigger on these branches
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'           # Daily at 2 AM UTC

jobs:
  test:                           # Job name
    runs-on: ubuntu-latest        # Runner OS/type
    strategy:
      matrix:                      # Run job multiple times with different configs
        python-version: ['3.9', '3.10', '3.11']
    
    steps:
      - uses: actions/checkout@v4 # Action: Clone repo
      
      - name: Set up Python       # Step name (appears in logs)
        uses: actions/setup-python@v5
        with:                     # Parameters for the action
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |                    # Multi-line shell command
          python -m pip install --upgrade pip
          pip install -r requirement.txt
      
      - name: Run tests
        run: pytest test_main.py
        
      - name: Upload coverage
        uses: codecov/codecov-action@v3  # Third-party action
        if: always()              # Run even if previous step failed
```

---

## Production-Level Implementation

### What Your f1_models Project Needs:

1. **CI Pipeline** (Already have basic version)
   - Run tests on every push/PR
   - Check code quality (linting, type checking)
   - Build artifacts

2. **Dependency Caching** (NEW)
   - Cache pip packages between runs
   - Reduces execution time from 2-3 min to 30-60 sec

3. **Code Quality** (NEW)
   - Code linting (flake8/pylint)
   - Type checking (mypy)
   - Code formatting checks (black)

4. **Security Scanning** (NEW)
   - Dependency vulnerability scanning
   - Secret scanning to prevent token leaks

5. **Performance Testing** (NEW)
   - Memory usage monitoring
   - Model inference latency testing

6. **Model Validation** (NEW)
   - Load trained models
   - Validate predictions
   - Performance regression detection

7. **Deployment** (NEW)
   - Auto-deploy to Render on main branch
   - Health checks for deployed service

8. **Notifications** (NEW)
   - Slack notifications for failures
   - PR comments with test results

---

## Best Practices for ML Projects

### 1. **Cache Everything Possible**
```yaml
- uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirement.txt') }}
    restore-keys: ${{ runner.os }}-pip-
```

### 2. **Use Matrix for Testing Multiple Versions**
```yaml
strategy:
  matrix:
    python-version: ['3.9', '3.10', '3.11']
    os: [ubuntu-latest, windows-latest]
```

### 3. **Fail Fast on Errors**
```yaml
steps:
  - name: Run linter
    run: pylint *.py --fail-under=8.0
```

### 4. **Separate Jobs by Concern**
- test job: unit tests, integration tests
- lint job: code quality checks
- security job: vulnerability scanning
- deploy job: only if tests pass

### 5. **Use Artifacts for ML Models**
```yaml
- name: Upload trained models
  uses: actions/upload-artifact@v3
  with:
    name: trained-models
    path: *.json
```

### 6. **Monitor Performance Metrics**
- Track model prediction time
- Monitor memory usage
- Detect performance regressions

### 7. **Use Environment Secrets**
```yaml
env:
  SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK }}
```

### 8. **Branch Protection Rules**
- Require all checks to pass before merge
- Require PR reviews
- Require status checks from Actions

---

## Workflow File Structure for f1_models

Your repository should have:

```
.github/
  workflows/
    01-ci.yml              # Core CI: tests, linting
    02-code-quality.yml    # Static analysis
    03-security.yml        # Dependency scanning
    04-deploy.yml          # Deployment to Render
    05-performance.yml     # Model performance testing
```

---

## Variables and Contexts

GitHub Actions provides built-in variables:

```yaml
- name: Print context info
  run: |
    echo "Event: ${{ github.event_name }}"
    echo "Branch: ${{ github.ref_name }}"
    echo "Commit: ${{ github.sha }}"
    echo "Actor: ${{ github.actor }}"
    echo "Run ID: ${{ github.run_id }}"
```

### Expressions:
```yaml
if: github.event_name == 'push' && github.ref == 'refs/heads/main'
if: success()  # Previous step succeeded
if: failure()  # Previous step failed
if: always()   # Run regardless
```

---

## Common Patterns

### 1. Deploy Only on Main Branch
```yaml
if: github.ref == 'refs/heads/main' && success()
```

### 2. Conditional Job Dependencies
```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    outputs:
      coverage: ${{ steps.test.outputs.coverage }}
  deploy:
    needs: test
    if: success()
```

### 3. Matrix Strategy with Exclude
```yaml
strategy:
  matrix:
    python: ['3.9', '3.10', '3.11']
    os: [ubuntu, windows]
  exclude:
    - python: '3.9'
      os: windows
```

### 4. Timeout and Retry
```yaml
- name: Run tests
  run: pytest
  timeout-minutes: 10
  continue-on-error: true  # Don't fail job
```

---

## Next Steps

1. Review the individual workflow files in `.github/workflows/`
2. Customize secrets and environment variables
3. Set up branch protection rules in Settings
4. Monitor Actions tab for logs and metrics
5. Integrate with Slack/Discord for notifications

---

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GitHub Actions Marketplace](https://github.com/marketplace?type=actions)
- [Workflow Syntax Reference](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
