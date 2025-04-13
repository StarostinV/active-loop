from setuptools import setup, find_packages

setup(
    name="active_loop",
    version="0.1.0",
    description="Active loop for fetching scan data",
    author="Vladimir Starostin",
    author_email="vladimir.starostin@uni-tuebingen.de",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.20.0",
        "click>=8.0.0",
        "plotext>=5.0.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
    ],
    entry_points={
        "console_scripts": [
            "active-fetch-server=active_loop.fetch_server:main",
            "active-fetch-client=active_loop.fetch_client:main",
            "active-push-server=active_loop.push_server:main",
            "active-push-client=active_loop.push_client:main",
            "active-dummy-ml=active_loop.dummy_ml:main",
            "active-run=active_loop.hydra_run:hydra_main",
            "active-consecutive=active_loop.consecutive_loop:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
