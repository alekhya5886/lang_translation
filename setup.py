from setuptools import setup, Extension

setup(
    ext_modules=[
        Extension("IndicTransToolkit.processor", ["IndicTransToolkit/processor.c"])
    ]
)
