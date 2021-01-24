from setuptools import find_packages, setup

requirements = [
    "numpy>=1.16",
    "backends",
    "varz>=0.5.3",
    "stheno",
    "wbml",
    "jax",
    "jaxlib",
]

setup(
    packages=find_packages(exclude=["docs"]),
    python_requires=">=3.6",
    install_requires=requirements,
    include_package_data=True,
)
