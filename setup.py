from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    requirements = []

    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name="support_ticket_ai",
    version="0.0.1",
    author="Murali Krishna",
    author_email="muralikrishana1729@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)