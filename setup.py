"""
CPU大模型训练框架安装脚本
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="cpu-llm-framework",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="CPU优化的Transformer大模型训练框架",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/transformer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "psutil>=5.9.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "tensorboard": [
            "tensorboard>=2.12.0",
        ],
        "transformers": [
            "transformers>=4.30.0",
            "tokenizers>=0.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pretrain=scripts.pretrain:main",
            "finetune=scripts.finetune:main",
            "generate=scripts.generate:main",
        ],
    },
)
