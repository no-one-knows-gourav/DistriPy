from setuptools import setup, find_packages

try:
    with open("README.md", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = ""

setup(
    name='DistriPy',
    version='0.1.0',
    author='Saigourav Sahoo',
    author_email='sgsahoo77@gmail.com',
    description='Symbolic and numerical probability library for continuous random variables and stochastic processes',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/no-one-knows-gourav/DistriPy',
    packages=find_packages(),
    install_requires=[
        'sympy',
        'numpy',
        'matplotlib',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    python_requires='>=3.7',
    license='MIT',
    keywords='probability statistics stochastic symbolic mathematics finance',
)