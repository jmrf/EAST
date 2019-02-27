import os
import subprocess
from distutils.command.build import build as _build
from distutils.core import setup, Extension


install_requires = [
    "Shapely==1.5.13",
    "numpy==1.13.0",
    "Flask==0.12.2",
    "plumbum==1.6.2",
    "matplotlib==2.2.3",
    "opencv_python==3.3.1.11",
    "tensorflow==1.10.0"
]


def cmake_build():
    w_dir = os.path.join(os.getcwd(), "east", "lanms")
    if subprocess.call("make --directory {}".format(w_dir), shell=True) != 0:
        raise EnvironmentError("error calling make")


class BuildWithCmake(_build):

    def run(self):
        cmake_build()
        # can't use super() here because _build is an old style class in 2.7
        _build.run(self)


setup(
    name='east',
    version='0.1dev',
    author='jmrf',
    author_email='jose.marcos.rf@gmail.com',
    license='GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007',
    long_description=open('README.md').read(),
    packages=['east', 'east/nets', 'east/lanms'],
    package_data={'': ['east/lanms/adaptor.so']},
    install_requires=install_requires,
    extras_require={
        "libgeos-dev": ["libgeos-dev"],
    },
    cmdclass={
        'build': BuildWithCmake,
    }
)
