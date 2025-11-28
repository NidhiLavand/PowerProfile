from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="energy-meter",
    version="0.1.0",
    author="Nidhi Lavand",
    author_email="nidhilavand@gmail.com",
    description="Measure CPU/GPU energy, power, and carbon emissions for ML workloads.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "psutil",
        "pynvml",
        "torch",
    ],
    python_requires=">=3.8",
)

