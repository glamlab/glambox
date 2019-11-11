import codecs
import os
import sys

from setuptools import find_packages, setup

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
README_FILE = os.path.join(PROJECT_ROOT, "README.md")
VERSION_FILE = os.path.join(PROJECT_ROOT, "glambox", "version.py")
REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, "requirements.txt")


def get_long_description():
    with codecs.open(README_FILE, "rt") as buff:
        return buff.read()


def get_requirements():
    with codecs.open(REQUIREMENTS_FILE) as buff:
        return buff.read().splitlines()


with open(VERSION_FILE) as buff:
    exec(buff.read())

if len(set(("test", "easy_install")).intersection(sys.argv)) > 0:
    import setuptools

tests_require = []
extra_setuptools_args = {}

setup(
    name="glambox",
    version=__version__,
    description="GLAMbox: A toolbox to fit the Gaze-weighted Linear Accumulator Model",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="http://github.com/glamlab/glambox",
    download_url="https://github.com/glamlab/glambox/archive/%s.tar.gz" % __version__,
    install_requires=get_requirements(),
    maintainer="Felix Molter <felixmolter@gmail.com>, Armin W. Thomas <athms.research@gmail.com>",
    maintainer_email="glambox.berlin@gmail.com",
    packages=find_packages(exclude=["tests", "test_*"]),
    tests_require=tests_require,
    license="MIT",
    **extra_setuptools_args,
)