from setuptools import setup, find_packages

setup(
    name="IndicTransToolkit",
    version="0.1.0",
    packages=find_packages(),  # This finds all your Python packages and modules
    install_requires=[
        "regex",
        "tqdm",
        "indic-nlp-library",
        "sacremoses",
        # add other dependencies your processor.py needs
    ],

)
