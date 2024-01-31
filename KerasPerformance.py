import subprocess
import logging
from timeit import timeit

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, filename='keras_performance.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

def run_command(command):
    try:
        subprocess.check_call(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        logger.critical(f"Error executing command: {command}")
        logger.critical(e.output.decode())

def create_venv(keras_version):
    logger.info(f"Creating venv: {keras_version}")
    PACAKGE_PATH = "./packages/"
    requirements = "numpy pandas os abc timeit sklearn"
    run_command(f"python -m venv {PACAKGE_PATH}{keras_version}")
    [f"{venv1}\\Scripts\\python.exe", "-m", "pip", "install", "package==version1"]
    run_command(f"source {PACAKGE_PATH}{keras_version}/bin/activate && pip install keras=={keras_version} {requirements}")
    logger.info("venv created")

def run_version_test(keras_version):
    logger.info(f"Running tests on {keras_version}")
    run_command(f"source ./{keras_version}/bin/activate && python test_keras_version.py")
    logger.info("Tests completed")

def main():
    # Params
    keras_versions = ["2.8.0", "2.9.0"]  # , "2.10.0", "2.11.0", "2.12.0", "2.13.0", "2.14.0", "2.15.0", "3.0.0"]
    for keras_version in keras_versions:
        create_venv(keras_version)
    for keras_version in keras_versions:
        run_version_test(keras_version)

if __name__ == "__main__":
    main()