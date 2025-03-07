from setuptools import find_packages, setup

__repository_url__ = "https://github.com/lifeiteng/OmniSenseVoice"
__download_url__ = "https://github.com/lifeiteng/OmniSenseVoice/releases"


setup(
    name="OmniSenseVoice",
    version="0.2.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"": []},
    description="OmniSenseVoice",
    author="lifeiteng0422@gmail.com",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    # The project's main homepage.
    url=__repository_url__,
    download_url=__download_url__,
    readme="README.md",
    python_requires=">=3.8",
    install_requires=[
        "funasr_onnx==0.4.1",
        "modelscope==1.18.0",
        "funasr==1.1.6",
        "lhotse>=1.24.2",
        "kaldialign",
        "torch",
        "torchaudio",
    ],
    entry_points={
        "console_scripts": ["omnisense=omnisense.bin:cli"],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
)
