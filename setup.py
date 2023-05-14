from setuptools import setup

setup(
    name="app-example",
    version="0.0.1",
    author="Vita",
    author_email="orehovvv2007@gmail.com",
    description="FastApi medical analytics",
    install_requires=[
        "fastapi==0.95.1",
        "uvicorn[standard]==0.22.0",
        "SQLAlchemy==2.0.13",
        "uuid==1.30",
        "easyocr==1.6.2",
        "psycopg2-binary==2.9.6",
        "python-multipart==0.0.6"
    ],
    scripts=['./main.py']
)