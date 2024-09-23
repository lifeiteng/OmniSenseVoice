from setuptools import find_packages, setup

setup(
    name="OmniSenseVoice",
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"": []},
    description="OmniSenseVoice",
    author="lifeiteng0422@gmail.com",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    python_requires=">=3.8",
    install_requires=[
        "funasr_onnx==0.4.1",
        "modelscope==1.18.0",
        "funasr==1.1.6",
        "lhotse>=1.24.2",
    ],
    entry_points={
        "console_scripts": ["omnisense=omnisense.bin:cli"],
    },
    include_package_data=True,
)
