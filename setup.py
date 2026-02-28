from setuptools import find_packages, setup

setup(
    name="coffee_roast_ai",
    version="0.1.0",
    author="Mrudula",
    description="A modular deep learning pipeline for coffee roast classification",
    # This automatically finds the 'src' folder and treats it as a package
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "tensorflow",
        "pandas",
        "kagglehub",
        "pyyaml",
        "pillow",
        "streamlit",
        "matplotlib"
    ],
python_requires=">=3.8",
)