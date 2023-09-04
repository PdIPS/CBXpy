import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cbx", 
    version="0.1.0",
    author="Tim Roith",
    author_email="tim.roith@fau.de",
    description="CBXpy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PdIPS/CBXpy",
    packages=setuptools.find_packages(),
    classifiers=[
                "Programming Language :: Python :: 3",
                "License :: OSI Approved :: MIT License",
                "Operating System :: OS Independent"],
    install_requires=[  'numpy', 
                        'scipy', 
                        'scikit-learn', 
                        'matplotlib'],
    python_requires='>=3.6',
)
