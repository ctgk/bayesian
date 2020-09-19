from setuptools import setup


__version__ = '0.0.1'


install_requires = [
    'numpy',
]
app_requires = [
    'matplotlib',
    'streamlit',
    'streamlit_drawable_canvas',
]
develop_requires = [
    'autopep8',
    'flake8',
    'pep8-naming',
    'pre-commit',
    'pytest',
]


setup(
    name='bayesian',
    version=__version__,
    author='ctgk',
    author_email='r1135nj54w@gmail.com',
    description='Bayesian models',

    python_requires='>=3',
    install_requires=install_requires,
    extras_require={
        'app': app_requires,
        'develop': app_requires + develop_requires,
    },

    zip_safe=False,
)
