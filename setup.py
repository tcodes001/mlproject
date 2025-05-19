from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    '''
    This function reads a requirements file and returns a list of required packages.
    '''
    requirements=[]
    HYPEN_E_DOT= '-e .'
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
name='mlproject',
version='0.0.1',
author='Tanisha',
author_email='tanishasolanki26@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')

)