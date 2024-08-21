from __future__ import annotations

from typing import Optional

from typeguard import typechecked

from mpqp import Language
from mpqp.core.circuit_photonic import PhotonicCircuit
from mpqp.core.instruction.measurement import ComputationalBasis
from mpqp.core.instruction.measurement.basis_measure import BasisMeasure
from mpqp.core.instruction.measurement.expectation_value import ExpectationMeasure
from mpqp.execution.devices import QUANDELADevice
from mpqp.execution.job import Job, JobType
from mpqp.execution.result import Result, Sample, StateVector


@typechecked
def run_quandela(job: Job) -> Result:
    """Executes the job on the right Quandela device precised in the job in
    parameter.

    Args:
        job: Job to be executed.

    Returns:
        A Result after submission and execution of the job.
    
    Note:
        This function is not meant to be used directly, please use
        :func:``run<mpqp.execution.runner.run>`` instead.
    """
    return run_local(job) if not job.device.is_remote() else run_quandela_remote(job)


@typechecked
def run_quandela_remote(job: Job) -> Result:
    """
    Executes the job remotely on a Quandela quantum device. 

    Args:
        job: job to be executed.

    Returns:
        Result: The result after submission and execution of the job.

    Raises:
        ValueError: If the job's device is not an instance of QUANDELADevice.
        NotImplementedError: If the job's device is not supported.
        NotImplementedError: If the job type or basis measure is not supported.
    """
    raise NotImplementedError(
        f"{job.device} is not handled for the moment."
    )

    return extract_result(result_sim, job, job.device)


@typechecked
def run_local(job: Job) -> Result:
    """
    Executes the job locally.

    Args:
        job : The job to be executed.

    Returns:
        Result: The result after submission and execution of the job.

    Raises:
        ValueError: If the job device is not QUANDELADevice.
    """
    assert isinstance(job.device, QUANDELADevice)
    assert  isinstance(job.circuit, PhotonicCircuit)
    
    job_cirq_circuit = job.circuit.to_other_language(Language.PERCEVAL)
    assert isinstance(job_cirq_circuit, Cirq_circuit)

    simulator = Simulator(noise=None)

    if job.job_type == JobType.STATE_VECTOR:
        
    elif job.job_type == JobType.SAMPLE:
        assert isinstance(job.measure, BasisMeasure)
        if isinstance(job.measure.basis, ComputationalBasis):
            
        else:
            raise NotImplementedError(
                "Does not handle other basis than the ComputationalBasis for the moment"
            )
    elif job.job_type == JobType.OBSERVABLE:
        assert isinstance(job.measure, ExpectationMeasure)

    else:
        raise ValueError(f"Job type {job.job_type} not handled")

    return extract_result(result_sim, job, job.device)


def extract_result(
    result: (
        StateVectorTrialResult
        | cirq_result
        | list[float]
        | list[ObservableMeasuredResult]
    ),
    job: Optional[Job] = None,
    device: Optional[QUANDELADevice] = None,
) -> Result:
    """Extracts the needed data from ``cirq`` result and packages it into a
    ``MPQP`` :class:`Result<mpqp.execution.result.Result>`.

    Args:
        result : The result of the simulation.
        job : The original job. Defaults to None.
        device : The device used for the simulation. Defaults to None.

    Returns:
        Result: The formatted result.

    Raises:
        NotImplementedError: If the job is None or the type is not supported.
        ValueError: If the result type does not match the expected type for the
            job type.
    """
    if job is None:
        raise NotImplementedError("result from job None is not implemented")
    else:
        if job.job_type == JobType.SAMPLE:
            if not isinstance(result, cirq_result):
                raise ValueError(
                    f"result: {type(result)}, must be a cirq_result for job type {job.job_type}"
                )
            return extract_result_SAMPLE(result, job, device)
        elif job.job_type == JobType.STATE_VECTOR:
            if not isinstance(result, StateVectorTrialResult):
                raise ValueError(
                    f"result: {type(result)}, must be a cirq_result for job type {job.job_type}"
                )
            return extract_result_STATE_VECTOR(result, job, device)
        elif job.job_type == JobType.OBSERVABLE:
            if isinstance(result, cirq_result):
                raise ValueError(
                    f"result: {type(result)}, must be a cirq_result for job type {job.job_type}"
                )
            return extract_result_OBSERVABLE(result, job, device)
        else:
            raise NotImplementedError("Job type not supported")


def extract_result_SAMPLE(
    result: cirq_result,
    job: Job,
    device: Optional[QUANDELADevice] = None,
) -> Result:
    """
    Extracts the result from a sample-based job.

    Args:
        result : The result of the simulation.
        job : The original job.
        device : The device used for the simulation. Defaults to None.

    Returns:
        Result: The formatted result.
    """
    nb_qubits = job.circuit.nb_qubits

    keys_in_order = sorted(result.records.keys())
    counts = result.multi_measurement_histogram(keys=keys_in_order)

    data = [
        Sample(
            bin_str="".join(map(str, state)),
            probability=count / sum(counts.values()),
            nb_qubits=nb_qubits,
        )
        for (state, count) in counts.items()
    ]

    shot = job.measure.shots if job.measure is not None else 0
    return Result(job, data, None, shot)


def extract_result_STATE_VECTOR(
    result: StateVectorTrialResult,
    job: Job,
    device: Optional[QUANDELADevice] = None,
) -> Result:
    """
    Extracts the result from a state vector-based job.

    Args:
        result : The result of the simulation.
        job : The original job.
        device : The device used for the simulation. Defaults to None.

    Returns:
        Result: The formatted result.
    """
    state_vector = result.final_state_vector
    state_vector = StateVector(
        state_vector, job.circuit.nb_qubits, state_vector_to_probabilities(state_vector)
    )
    return Result(job, state_vector, 0, 0)


def extract_result_OBSERVABLE(
    result: list[float] | list[ObservableMeasuredResult],
    job: Job,
    device: Optional[QUANDELADevice] = None,
) -> Result:
    """
    Extracts the result from an observable-based job.

    Args:
        result : The result of the simulation.
        job : The original job.
        device : The device used for the simulation. Defaults to None.

    Returns:
        Result: The formatted result.
    """
    mean = 0.0
    variance = 0.0
    if job.measure is None:
        raise NotImplementedError("job.measure is None")
    for result1 in result:
        if isinstance(result1, float) or isinstance(result1, complex):
            mean += abs(result1)
        if isinstance(result1, ObservableMeasuredResult):
            mean += result1.mean
            # 3M-TODO variance not supported variance += result1.variance
    return Result(job, mean, variance, job.measure.shots)
