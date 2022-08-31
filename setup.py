from setuptools import setup, find_packages


setup(
    name="pygrad",
    version="0.0.1",
    author="Kazi Jawad",
    url="https://github.com/kazijawad/pygrad",
    packages=find_packages(exclude=("demo")),
    python_requires=">=3.10"
)
