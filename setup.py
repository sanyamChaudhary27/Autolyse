from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="autolyse",
    version="0.1.0",
    author="Your Name",
    description="Auto EDA with AI insights - Generate comprehensive exploratory data analysis with 2 lines of code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sanyamChaudhary27/Autolyse",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "plotly>=5.0.0",
        "scikit-learn>=1.0.0",
        "google-generativeai>=0.3.0",
        "Jinja2>=3.0.0",
        "scipy>=1.8.0",
    ],
)
