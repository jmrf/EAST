import os
import subprocess
from distutils.core import setup
from distutils.command.build import build as _build
from setuptools import find_packages, findall
from pprint import pformat


def echo(msg):
    print("{0}\n{1}\n{0}".format("-" * 120, msg))


def subdir_findall(dir, subdir):
    strip_n = len(dir.split('/'))
    path = '/'.join((dir, subdir))
    return ['/'.join(s.split('/')[strip_n:]) for s in findall(path)]


def cmake_build():
    w_dir = os.path.join(os.getcwd(), "east", "lanms")
    if subprocess.call("make --directory {}".format(w_dir), shell=True) != 0:
        raise EnvironmentError("error calling make")
    else:
        echo("LANMS: should now be compiled "
             "in '{}': {}".format(w_dir, pformat(os.listdir(w_dir))))


class BuildWithCmake(_build):

    def run(self):
        cmake_build()
        # can't use super() here because _build is an old style class in 2.7
        _build.run(self)


install_requires = [
    "Shapely==1.5.13",
    "numpy==1.13.0",
    "Flask==0.12.2",
    "plumbum==1.6.2",
    "matplotlib==2.2.3",
    "opencv_python==3.3.1.11",
    "tensorflow==1.10.0"
]

packages = find_packages(exclude=['tests', 'training_samples'])
packages += ['east/lanms/include/']
echo(packages)

east_package_data = subdir_findall('east', 'lanms') + \
    subdir_findall('east', 'static') + \
    subdir_findall('east', 'templates')


setup(
    name='east',
    version='0.1dev',
    author='jmrf',
    author_email='jose.marcos.rf@gmail.com',
    license='GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007',
    long_description=open('README.md').read(),
    extras_require={
        "libgeos-dev": ["libgeos-dev"],
    },
    packages=packages,
    package_data={
        'east': east_package_data
    },
    cmdclass={
        'build': BuildWithCmake,
    },
    include_package_data=True,
    install_requires=install_requires,
)
