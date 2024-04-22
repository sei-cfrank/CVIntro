# HandsOnCV
Repo for CMU-SEI AI Division "Hands-On Computer Vision" course

## Virtual environment setup from command line
1. cd into the top level of this repo
2. Create a virtual environment called "venv" with the command: `python -m venv venv`
3. Activate this virtual environment before installing any packages. on Windows activate with `.\venv\Scripts\activate` alternatively on either macOS or Linux activate with `source venv/bin/activate`.
4. Your command line prompt should now indicate that the venv virtual environment is activated.
5. Now that the virtual environment is activated, install any neccesary packages using `pip install`
6. venv/ is included in the gitignore file, so any changes there will not be tracked by Git
7. Instead to ensure that others who clone the repo can install the same dependencies, save them to a requirements.txt file: `pip freeze > requirements.txt`
8. Commit and push any changes to the requirements.txt to GitHub
9. All of the packages listed in the requirements.txt file can be installed at once using `pip install -r requirements.txt`
