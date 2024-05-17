from getpass import getpass
from termcolor import colored

import cirq_ionq as ionq
from cirq_ionq.ionq_exceptions import IonQException
from mpqp.execution.connection.env_manager import get_env_variable, save_env_variable


def test_ionq_connection() -> bool:
    """
    Test the connection to the IonQ service.

    Returns:
        bool: True if the connection is successful, False otherwise.
    """
    service = ionq.Service(default_target="simulator")
    try:
        service.list_jobs()
        return True
    except IonQException:
        print(colored("Wrong credentials", "red"))
        return False


def config_ionq_key():
    """
    Configure the IonQ account by setting the API token.

    Returns:
        tuple: A message indicating the result of the configuration and an empty list.
    """
    was_configured = get_env_variable("GOOGLE_IONQ_CONFIGURED") == "True"

    if was_configured:
        decision = input(
            "An IONQ key is already configured. Do you want to update it? [y/N]"
        )
        if decision.lower().strip() != "y":
            return "Canceled.", []

    token = getpass("Enter your ionq token (hidden): ")
    if token == "":
        print(colored("Empty credentials", "red"))
        getpass("Press 'Enter' to continue")
        return "", []
    old_token = get_env_variable("IONQ_API_KEY")
    save_env_variable("IONQ_API_KEY", token)
    if test_ionq_connection():
        save_env_variable("GOOGLE_IONQ_CONFIGURED", "True")
        return "IONQ key correctly configured", []
    else:
        if was_configured:
            save_env_variable("IONQ_API_KEY", old_token)
        else:
            save_env_variable("GOOGLE_IONQ_CONFIGURED", "False")
        getpass("Press 'Enter' to continue")
        return "", []


def get_google_account_info() -> str:
    """
    Get the IonQ API key from the environment variables.

    Returns:
        str: A string containing the IonQ API key.
    """
    ionq_api_key = get_env_variable("IONQ_API_KEY")

    return f"""   IONQ_API_KEY : '{ionq_api_key}'"""


def get_all_job_ids() -> list[str]:
    """
    Retrieves all job IDs associated with IonQ jobs.

    Returns:
        A list of job IDs.
    """
    ionq_job_ids = []
    if get_env_variable("IONQ_API_KEY") == "True":
        service = ionq.Service()
        ionq_job_ids = [job.job_id() for job in service.list_jobs()]
    return ionq_job_ids
