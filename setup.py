from setuptools import setup, find_packages
from pathlib import Path

description = ['visloc_localization']

root = Path(__file__).parent
with open(str(root / 'README.md'), 'r', encoding='utf-8') as f:
    readme = f.read()

with open(str(root / '_version.py'), 'r') as f:
    version = eval(f.read().split('__version__ = ')[1].split()[0])

with open(str(root / 'requirements.txt'), 'r') as f:
    dependencies = f.read().split('\n')


    
setup(
    name='visloc_localization',
    version=version,
    python_requires='>=3.7',
    author='Tarek BOUAMER',
    author_email="tarekbouamer1788@gmail.com",
    description=description,
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/Tarekbouamer',
    
    # pkgs
    packages=find_packages(),

    # install  
    install_requires=dependencies,

    dependency_links=[
        # 'git+https://github.com/Tarekbouamer/visloc_retrieval#egg=visloc_retrieval',
        'git+https://github.com/jenicek/asmk#egg=asmk',
        'git+https://github.com/stevenlovegrove/Pangolin#egg=pangolin'
        ]

)
