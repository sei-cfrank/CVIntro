# HandsOnCV
Repo for CMU-SEI AI Division "Hands-On Computer Vision" course for Gauntlet Delivery (No Pis)

Streamline
Binder Deliver:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sei-cfrank/CVIntro/HEAD)


## Virtual environment setup from command line
1. cd into the top level of this repo
2. Create a virtual environment called "venv" with the command: `python -m venv --system-site-packages venv`
3. Activate this virtual environment before installing any packages. on Windows activate with `.\venv\Scripts\activate` alternatively on either macOS or Linux activate with `source venv/bin/activate`.
4. Your command line prompt should now indicate that the venv virtual environment is activated.
5. Now that the virtual environment is activated, install any neccesary packages using `pip install`
6. venv/ is included in the gitignore file, so any changes there will not be tracked by Git
7. Instead to ensure that others who clone the repo can install the same dependencies, save them to a requirements.txt file: `pip freeze > requirements.txt`
8. Commit and then push any changes to the requirements.txt to GitHub
9. All of the packages listed in the requirements.txt file can be installed at once using `pip install -r requirements.txt`
10. To deactivate the virtual environment simply run the command `deactivate`

## Setting up a Jupyter kernel
1. First make sure your virtual environment in activated and you have installed the system requirements as detailed above with `pip install -r requirements.txt`
2. On the command line in the top level of this repo run the follwing three commands
    - `ipython kernel install --user --name=venv`
    - `python -m ipykernel install --user --name=venv`
    - `python -m bash_kernel.install`
3. Start Jupyter notebook with `jupyter notebook`, a browser window with the Jupyter interface should open
4. Open a notebook (i.e. a .ipynb file) in that Jupyter browser window
5. Click on the "Kernel" menu
6. Choose "Change Kernel" and select the virtual environment kernel you created called "venv"
This ensures that your Jupyter notebooks is using the Python environment from your virtual environment
