from setuptools import find_packages, setup
from typing import List


def get_requirements() -> List[str]:
    """
    This function will return a list of requirements.
    """
    requirement_list: List[str] = []
    # Code to read the requirements.txt file and append to requirements_list variable
    with open("requirements.txt", 'r') as f:
        requirements = f.readlines()
        if requirements:
            for requirement in requirements:
                if requirement != "\n":
                    requirement_list.append(requirement.split('\n')[0])
    return requirement_list


setup(
    name="sensor",
    version="0.0.1",
    author="Naveen",
    author_email="naveenfaster@gmail.com",
    packages=find_packages(),
    install_requires=[],
)
