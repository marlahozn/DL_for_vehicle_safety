from setuptools import setup, find_packages

setup(
    name='paper_code',
    version='0.1.0',
    python_requires=">=3.7",
    packages=find_packages(include=['code_for_paper']),
    install_requires=[
        "pandas==1.3.5",
        "numpy==1.21.6",
        "matplotlib==3.5.3",
        "scikit-learn==1.0.2",
        "tensorflow==2.11.0",
        "PyYAML==6.0.1",
    ]
)