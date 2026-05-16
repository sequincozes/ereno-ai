from copy import deepcopy
from typing import Any


def validate_and_clamp_attack_config(
    candidate_config: dict[str, Any],
    baseline_config: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """
    Valida e corrige configurações degeneradas.

    O objetivo é impedir que a LLM reduza demais a intensidade ou frequência
    do ataque, o que poderia causar uma queda artificial do F1-score apenas
    porque o ataque ficou fraco demais ou raro demais.
    """

    config = deepcopy(candidate_config)
    warnings: list[str] = []

    # fault.prob não pode cair demais em relação ao baseline.
    baseline_fault_prob = get_nested(baseline_config, "fault.prob")
    candidate_fault_prob = get_nested(config, "fault.prob")

    if isinstance(baseline_fault_prob, (int, float)) and isinstance(candidate_fault_prob, (int, float)):
        min_fault_prob = baseline_fault_prob * 0.5

        if candidate_fault_prob < min_fault_prob:
            set_nested(config, "fault.prob", min_fault_prob)
            warnings.append(
                f"fault.prob ajustado de {candidate_fault_prob} para {min_fault_prob}, "
                "pois estava reduzindo demais a frequência do ataque."
            )

    # analog.deltaAbs.max não pode virar quase zero.
    baseline_delta_max = get_nested(baseline_config, "analog.deltaAbs.max")
    candidate_delta_max = get_nested(config, "analog.deltaAbs.max")

    if isinstance(baseline_delta_max, (int, float)) and isinstance(candidate_delta_max, (int, float)):
        min_delta_max = baseline_delta_max * 0.4

        if candidate_delta_max < min_delta_max:
            set_nested(config, "analog.deltaAbs.max", min_delta_max)
            warnings.append(
                f"analog.deltaAbs.max ajustado de {candidate_delta_max} para {min_delta_max}, "
                "pois estava tornando a perturbação física fraca demais."
            )

    # analog.deltaAbs.min precisa ser coerente com max.
    delta_min = get_nested(config, "analog.deltaAbs.min")
    delta_max = get_nested(config, "analog.deltaAbs.max")

    if isinstance(delta_min, (int, float)) and isinstance(delta_max, (int, float)):
        if delta_min > delta_max:
            set_nested(config, "analog.deltaAbs.min", delta_max)
            warnings.append(
                f"analog.deltaAbs.min ajustado de {delta_min} para {delta_max}, "
                "pois estava maior que analog.deltaAbs.max."
            )

    # trapArea.multiplier.max não pode ficar praticamente normal.
    baseline_trap_max = get_nested(baseline_config, "trapArea.multiplier.max")
    candidate_trap_max = get_nested(config, "trapArea.multiplier.max")

    if isinstance(baseline_trap_max, (int, float)) and isinstance(candidate_trap_max, (int, float)):
        min_trap_max = max(1.2, baseline_trap_max * 0.5)

        if candidate_trap_max < min_trap_max:
            set_nested(config, "trapArea.multiplier.max", min_trap_max)
            warnings.append(
                f"trapArea.multiplier.max ajustado de {candidate_trap_max} para {min_trap_max}, "
                "pois estava aproximando demais o ataque do tráfego normal."
            )

    # trapArea.multiplier.min precisa ser coerente com max.
    trap_min = get_nested(config, "trapArea.multiplier.min")
    trap_max = get_nested(config, "trapArea.multiplier.max")

    if isinstance(trap_min, (int, float)) and isinstance(trap_max, (int, float)):
        if trap_min > trap_max:
            set_nested(config, "trapArea.multiplier.min", trap_max)
            warnings.append(
                f"trapArea.multiplier.min ajustado de {trap_min} para {trap_max}, "
                "pois estava maior que trapArea.multiplier.max."
            )

    # trapArea.spikeProb não pode cair quase para zero.
    baseline_spike_prob = get_nested(baseline_config, "trapArea.spikeProb")
    candidate_spike_prob = get_nested(config, "trapArea.spikeProb")

    if isinstance(baseline_spike_prob, (int, float)) and isinstance(candidate_spike_prob, (int, float)):
        min_spike_prob = baseline_spike_prob * 0.4

        if candidate_spike_prob < min_spike_prob:
            set_nested(config, "trapArea.spikeProb", min_spike_prob)
            warnings.append(
                f"trapArea.spikeProb ajustado de {candidate_spike_prob} para {min_spike_prob}, "
                "pois estava reduzindo demais os eventos de ataque."
            )

    # Probabilidades devem ficar entre 0 e 1.
    probability_fields = [
        "fault.prob",
        "trapArea.spikeProb",
    ]

    for field in probability_fields:
        value = get_nested(config, field)

        if isinstance(value, (int, float)):
            clamped = min(max(value, 0.0), 1.0)

            if clamped != value:
                set_nested(config, field, clamped)
                warnings.append(
                    f"{field} ajustado de {value} para {clamped}, "
                    "pois probabilidades devem estar entre 0 e 1."
                )

    # Evita desligar incremento de StNum se for parte central do baseline.
    baseline_increment = get_nested(baseline_config, "incrementStNumOnFault")
    candidate_increment = get_nested(config, "incrementStNumOnFault")

    if baseline_increment is True and candidate_increment is False:
        set_nested(config, "incrementStNumOnFault", True)
        warnings.append(
            "incrementStNumOnFault foi restaurado para True para não descaracterizar o ataque baseline."
        )

    return config, warnings


def get_nested(data: dict[str, Any], path: str) -> Any:
    current: Any = data

    for part in path.split("."):
        if not isinstance(current, dict):
            return None

        if part not in current:
            return None

        current = current[part]

    return current


def set_nested(data: dict[str, Any], path: str, value: Any) -> None:
    current: Any = data
    parts = path.split(".")

    for part in parts[:-1]:
        if not isinstance(current, dict):
            return

        if part not in current:
            return

        current = current[part]

    if isinstance(current, dict):
        current[parts[-1]] = value