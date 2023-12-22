from setuptools import setup

setup(
    name="Alpha Zero Lib",
    version="1.0",
    description="Library for running and training Alpha Zero based models with backend in C++ and Rust.",
    author="Skyr",
    requires=[
        "numpy",
        "pygame",
        "gym~=0.26.2",
        "optuna",
        "tensorboard",
        "scikit-image",
        "torchviz",
        "torch>=2.0.0"
        "mysqlclient",
        "mysql-connector-python",
        "joblib",
        "dill",
        "ipython"
    ]

)
