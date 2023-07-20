import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

data_files_to_include = [('', ['README.md', 'LICENSE'])]

setuptools.setup(
    name='randmat',
    url="https://github.com/pfleig/randmat",
    author="Philipp Fleig",
    author_email="philipp.fleig@gmx.de",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version='0.1.0',
    description='Python package for random matrix computations.',
    license="GNU GPLv3",
    python_requires='>=3.5',
    install_requires = [
        "numpy",
        "scikit-learn",
        "scipy",
        "matplotlib",
    ],
    packages=setuptools.find_packages(),
    data_files = data_files_to_include,
    include_package_data=True,
)
