"""Unit test suite verifying Phase 0 scaffolding, documentation, and configs."""

from pathlib import Path
import yaml


REQUIRED_DOCUMENTS = [
    "AGENTS.md",
    "PROJECT_SPEC.md",
    "ARCHITECTURE.md",
    "RESEARCH_DECISIONS.md",
    "DATASET_GUIDE.md",
    "EXPERIMENT_PLAN.md",
    "IMPLEMENTATION_PLAN.md",
    "PROJECT_STATUS.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    ".gitignore",
]

REQUIRED_CONFIGS = [
    "configs/base.yaml",
    "configs/data.yaml",
    "configs/model.yaml",
    "configs/training.yaml",
    "configs/evaluation.yaml",
]

REQUIRED_DIRS = [
    "configs",
    "data/raw",
    "data/interim",
    "data/processed",
    "data/features",
    "data/manifests",
    "src/data",
    "src/models",
    "src/postprocessing",
    "src/risk",
    "src/evaluation",
    "src/utils",
    "scripts/data",
    "tests",
]


def test_required_documents_exist():
    """Verify that all Phase 0 required documents exist and have non-zero size."""
    root = Path(".")
    for doc in REQUIRED_DOCUMENTS:
        p = root / doc
        assert p.is_file(), f"Missing required document: {doc}"
        assert p.stat().st_size > 0, f"Document is empty: {doc}"


def test_required_directories_exist():
    """Verify that all skeleton directories exist."""
    root = Path(".")
    for d in REQUIRED_DIRS:
        p = root / d
        assert p.is_dir(), f"Missing directory: {d}"


def test_yaml_configs_are_valid():
    """Verify that all YAML configuration files parse cleanly without syntax errors."""
    root = Path(".")
    for cfg in REQUIRED_CONFIGS:
        p = root / cfg
        assert p.is_file(), f"Missing config file: {cfg}"
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            assert isinstance(data, dict), f"Config {cfg} did not parse into a dict"
            assert len(data) > 0, f"Config {cfg} is empty"


def test_source_packages_have_init():
    """Verify that all src subpackages have an __init__.py file."""
    root = Path(".")
    subpackages = ["data", "models", "postprocessing", "risk", "evaluation", "utils"]
    for pkg in subpackages:
        init_file = root / "src" / pkg / "__init__.py"
        assert init_file.is_file(), f"Missing __init__.py in src/{pkg}"


if __name__ == "__main__":
    print("Running test_required_documents_exist()...")
    test_required_documents_exist()
    print("Running test_required_directories_exist()...")
    test_required_directories_exist()
    print("Running test_yaml_configs_are_valid()...")
    test_yaml_configs_are_valid()
    print("Running test_source_packages_have_init()...")
    test_source_packages_have_init()
    print("\nALL SCAFFOLDING TESTS PASSED SUCCESSFULLY!")
