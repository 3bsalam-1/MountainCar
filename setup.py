from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / 'README.md'
long_description = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ''

setup(
    name='mountaincar-qlearning',
    version='1.0.0',
    author='3bsalam-1',
    description='Q-Learning implementation for MountainCar environment',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/3bsalam-1/MountainCar',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.8',
    install_requires=[
        'gymnasium>=0.29.0',
        'numpy>=1.24.0',
        'matplotlib>=3.7.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'mountaincar-train=scripts.train:main',
            'mountaincar-eval=scripts.evaluate:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='reinforcement-learning q-learning mountaincar gymnasium openai',
)
