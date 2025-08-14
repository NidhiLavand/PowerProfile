from setuptools import setup, find_packages

setup(
    name="Power_Profile",
    version="0.1.0",
    author="Nidhi Lavand",
    author_email="nidhilavand@gmail.com",
    description="Estimate power usage, energy usage and carbon emissions of Python programs",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/NidhiLavand/PowerProfile.git",
    packages=find_packages(),
    install_requires=[
        "psutil>=5.9.0"
    ],
    python_requires=">=3.7",
)

