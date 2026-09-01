from adversarial_ids.interfaces.dashboard.studio.bridge import (
    ExperimentConfig,
    ExperimentJob,
)


def test_benign_generation_status_follows_structured_logs():
    job = ExperimentJob(ExperimentConfig())

    assert job.benign_generation_status() is None

    job._buffer.write(  # noqa: SLF001 - teste do adaptador de logs da interface
        "[GEN:BENIGN] AUSENTE | iniciando geração\n"
    )
    assert job.benign_generation_status() == "generating"

    job._buffer.write(  # noqa: SLF001 - teste do adaptador de logs da interface
        "[GEN:BENIGN] CRIADO | dataset pronto\n"
    )
    assert job.benign_generation_status() == "created"
