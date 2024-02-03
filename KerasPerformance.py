import subprocess
import logging
from timeit import timeit

logger = logging.getLogger("Keras_Performance")
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO,
                    filename='keras_performance.log',
                    filemode='a',
                    format = '%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s')

PACKAGE_PATH = ".\\packages\\"

def run_command(command):
    try:
        subprocess.check_call(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        logger.critical(f"Error executing command: {command}")

def create_venv(keras_version):
    logger.info(f"Creating venv: {keras_version}")

    create_venv = ["python", "-m", "venv", f"{PACKAGE_PATH}{keras_version}"]
    requirements = [f"{PACKAGE_PATH}{keras_version}\\Scripts\\python.exe", "-m", "pip", "install", f"keras=={keras_version}", "numpy", "pandas", "scikit-learn", f"tensorflow=={keras_version}"]
    activate_venv = [f"{PACKAGE_PATH}{keras_version}\\Scripts\\activate"]
    run_command(create_venv)
    run_command(activate_venv + ["&&"] + requirements)
    logger.info("venv created")

def run_version_test(keras_version):
    logger.info(f"Running tests on {keras_version}")
    run_command(f"{PACKAGE_PATH}{keras_version}\\Scripts\\activate && python test_keras_version.py")
    logger.info("Tests completed")

def main():
    # Params
    keras_versions = ["2.9.0"]  # , "2.10.0", "2.11.0", "2.12.0", "2.13.0", "2.14.0", "2.15.0", "3.0.0"]
    # for keras_version in keras_versions:
    #     create_venv(keras_version)
    for keras_version in keras_versions:
        run_version_test(keras_version)

if __name__ == "__main__":
    main()