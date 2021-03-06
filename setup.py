import setuptools

setuptools.setup(
    name="extract_facial",
    version="v3.1-alpha",
    author="Vineet Kulkarni",
    author_email="2vineetk@gmail.com",
    description="Extract facial features in Python such as the eyes and mouth from MTCNN's predictions and find whether eyes or mouths are closed/open.",
    packages=["extract_facial"],
    include_package_data=True,
    url = "https://github.com/vink246/extract_facial.git",
    download_url = "https://github.com/vink246/extract_facial/archive/v3.1-alpha.tar.gz",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.5',
    install_requires= ["opencv-python", "imutils", "numpy"]
)
