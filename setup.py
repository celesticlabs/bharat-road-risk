from setuptools import setup, find_packages

setup(
    name="bharat-road-risk",
    version="0.1.0",
    description="Predictive road safety intelligence for Indian roads",
    author="Celestic Labs",
    author_email="celesticlabs@gmail.com",
    url="https://github.com/celestic-labs/bharat-road-risk",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "ultralytics>=8.0.0",
        "transformers>=4.35.0",
        "peft>=0.7.0",
        "gradio>=4.0.0",
    ],
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
    ],
)