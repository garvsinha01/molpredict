from setuptools import setup, find_packages

setup(
    name="molpredict",
    version="1.0.0",
    author="Prajwal Khadka",
    author_email="prajwal@example.com",
    description="Molecular property prediction using deep learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/prajwalkkhadka-cpu/molpredict",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "rdkit>=2023.3.1",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scipy>=1.11.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "molpredict-train=scripts.train:main",
            "molpredict-eval=scripts.evaluate:main",
            "molpredict-predict=scripts.predict:main",
        ]
    },
)
