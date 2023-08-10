# Author: Kenta Nakamura <c60evaporator@gmail.com>
# Copyright (c) 2020-2021 Kenta Nakamura
# License: BSD 3 clause

from setuptools import setup
import hytunaflow


def _requires_from_file(filename):
    return open(filename).read().splitlines()

DESCRIPTION = "hytunaflow: Tool for cooperating hydra, optuna and mlflow"
NAME = 'hytunaflow'
AUTHOR = 'Masashi Ueda'
AUTHOR_EMAIL = 'masashi620@gmail.com'
URL = 'https://github.com/masashi2ueda/hytunaflow'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/masashi2ueda/hytunaflow'
VERSION = hytunaflow.__version__
PYTHON_REQUIRES = ">=3.9"

EXTRAS_REQUIRE = {
}

PACKAGES = [
    'hytunaflow'
]

CLASSIFIERS = [
]

setup(name=NAME,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=AUTHOR,
    maintainer_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    python_requires=PYTHON_REQUIRES,
    install_requires=_requires_from_file('requirements.txt'),
    extras_require=EXTRAS_REQUIRE,
    packages=PACKAGES,
    classifiers=CLASSIFIERS
    )