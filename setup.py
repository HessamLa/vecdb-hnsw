"""
VecDB - Minimal Viable Vector Database
"""

from setuptools import setup, find_packages
import os

# Read version from package
def get_version():
    version_file = os.path.join(
        os.path.dirname(__file__),
        "src", "python", "vecdb", "_version.py"
    )
    if os.path.exists(version_file):
        with open(version_file) as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"\'')
    return "0.1.0"

# Read long description from README
def get_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, encoding="utf-8") as f:
            return f.read()
    return ""

setup(
    name="vecdb",
    version=get_version(),
    author="VecDB Team",
    description="Minimal Viable Vector Database with HNSW indexing",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/vecdb",
    license="MIT",

    # Package configuration
    package_dir={"": "src/python"},
    packages=find_packages(where="src/python"),

    # Python version requirement
    python_requires=">=3.9",

    # Dependencies
    install_requires=[
        "numpy>=1.20",
    ],

    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "pytest-xdist",
            "black",
            "isort",
            "mypy",
        ],
    },

    # Entry points (if CLI is added later)
    entry_points={
        "console_scripts": [
            # "vecdb=vecdb.cli:main",  # Uncomment if CLI is added
        ],
    },

    # Classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],

    # Include package data
    include_package_data=True,
    zip_safe=False,
)
