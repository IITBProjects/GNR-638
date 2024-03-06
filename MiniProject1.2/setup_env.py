import os
import subprocess
from venv import EnvBuilder
import platform

VENV_NAME = 'venv'

# Function to setup the environment
def setup_venv():
    if not os.path.exists(VENV_NAME):
        print("Creating virtual environment...")
        builder = EnvBuilder(with_pip=True)
        builder.create(VENV_NAME)
    else:
        print(f"Virtual environment {VENV_NAME} already exists.")

    print("Installing Python requirements...")
    pip_location = f'{VENV_NAME}\Scripts\pip' if platform.system() == 'Windows' else f'{VENV_NAME}/bin/pip'
    subprocess.run(f'{pip_location} install -r requirements.txt', shell=True)

setup_venv()
