# CONTRIBUTING.md — Development Guidelines & Coding Standards

## 1. Principles of Collaboration
All contributors and AI agents working on this codebase must adhere to:
- **Reproducibility**: Experiments must be fully reproducible using configuration files (`configs/`) and deterministic seeds.
- **Modularity**: Code must be cleanly organized into specialized packages within `src/`. Avoid single monolithic scripts.
- **Type Safety**: Use Python type hints (`typing`) across all public functions and classes.
- **Academic Rigor**: Never fabricate, hardcode, or alter experimental results. Report metrics honestly as `TARGET`, `EXPECTED`, or `MEASURED`.

---

## 2. Environment & Dependency Rules
- Always work inside the project virtual environment: `.venv\Scripts\Activate.ps1`.
- Never run global `pip install`.
- Declare all core and development dependencies in `pyproject.toml`.
- When adding a dependency, document the exact rationale and ensure compatibility with Python 3.11 and the 4 GB GPU constraint.

---

## 3. Code Style & Standards
- **Formatting**: PEP 8 compliance via `black` (line length 100) and `ruff`.
- **Imports**: Group imports into standard library, third-party packages, and local modules.
- **Documentation**: Provide Google-style docstrings for all modules, classes, and public functions.
- **Device Management**: Always support flexible device allocation (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`). Never hardcode `cuda:0` without fallback.

---

## 4. Git & Data Safety
- NEVER commit raw datasets, video files (`*.mp4`, `*.avi`), image frames (`*.png`), zip archives, model checkpoints (`*.pth`), or feature caches (`*.pt`, `*.npy`).
- Ensure `git status` remains clean before committing. Only tracked source files, configurations, small metadata manifests, and tests should be committed.
- Commit messages should follow the Conventional Commits specification (e.g., `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
