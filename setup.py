"""Setup script for Vertex AI Trainer Package.

This setup.py file is used to package the training code for Vertex AI.
"""

from setuptools import find_packages, setup

REQUIRED_PACKAGES = [
    'torch>=2.0.0',
    'torchvision>=0.15.0',
    'tensorboard>=2.12.0',
    'google-cloud-aiplatform>=1.25.0',
    'cloudml-hypertune>=0.1.0.dev6',
    'numpy>=1.23.0',
]

setup(
    name='vertex_ai_trainer',
    version='1.0.0',
    description='LeNet5 Hyperparameter Optimization for Vertex AI',
    author='Dan Dye',
    author_email='dandye@@users.noreply.github.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
