from setuptools import setup, find_packages

setup(
    name="rd_plus_active_diagnostic",
    version="0.2.0",
    description="RD++ anomaly detection with active diagnostic pipeline",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "Pillow>=9.5.0",
        "opencv-python>=4.5.0",
        "scipy>=1.10.0",
        "scikit-image>=0.19.0",
        "matplotlib>=3.5.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.20.0",
        "pydantic>=2.0.0",
    ],
)
