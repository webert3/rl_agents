from setuptools import setup, find_packages

INSTALL_REQUIREMENTS = [
      "numpy==1.16.3",
      "tensorflow>=1.15.2",
      "gym==0.17.1",
      "jupyter==1.0.0"
]

setup(name="rl_agents",
      python_requires=">3.5",
      packages=find_packages(exclude=("tests", "tests.*")),
      include_package_data=True,
      description="RL algorithms for various tasks",
      author="Theodore Weber",
      author_email="weber.ted2@gmail.com",
      install_requires=INSTALL_REQUIREMENTS,
      zip_safe=False)

