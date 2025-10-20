from setuptools import setup,find_packages

with open('requirements.txt') as f:
    requirements=f.read().splitlines()

setup(
    name='An AI-Powered Dual Output System for Emotion-Aware Music and Sheet Composition using Text Prompts',
    version=0.1,
    author='Team 3',
    packages=find_packages(),
    install_requires=requirements
)