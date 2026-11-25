"""Repositório de traces dos experimentos: (ataque, config, seed) → CSV.

Por que existe
--------------
O loop gera um trace por rodada e o descarta no diretório da execução. Os
experimentos do artigo precisam de outra coisa: o **mesmo** trace reaproveitado
por várias estratégias e detectores (senão a comparação mistura o efeito da
estratégia com o da regeneração), e cada seed com **o próprio tráfego normal**.

Até 13/10/2026 todo trace usava o mesmo arquivo benigno — as linhas normais do
teste eram, byte a byte, as do treino, e um split "por seed" seria de fachada.
Aqui cada seed tem um benigno gerado pelo ``CREATE_BENIGN`` do ERENO com aquela
seed (ver o commit ``c36bee2`` no ERENO-2.0), e o trace de ataque da seed é
gerado sobre ele, com a mesma seed na RNG dos creators.

Cache
-----
A chave de um trace é o hash de (ataque, config canônica, seed, hash do JAR).
Trocar o JAR invalida o cache sozinho — um trace gerado por um gerador com bug
não pode responder por um gerador corrigido. Os traces ficam comprimidos
(``.csv.gz``; o pandas lê direto) com um ``.json`` ao lado dizendo de onde vieram.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

from adversarial_ids.config.attacks_registry import get_attack_spec
from adversarial_ids.config.settings import (
    ERENO_SOURCE_DIR,
    GENERATOR_ACTION_CONFIG_RELATIVE_PATH,
    GENERATOR_BENIGN_ACTION_CONFIG_RELATIVE_PATH,
    GENERATOR_JAR_PATH,
    GENERATOR_MAX_RETRIES,
    GENERATOR_OUTPUT_DATASET_PATH,
    GENERATOR_RETRY_BACKOFF_SECONDS,
    GENERATOR_RUN_COMMAND,
    GENERATOR_RUNTIME_DIR,
    GENERATOR_TIMEOUT_SECONDS,
    OUTPUTS_DIR,
)
from adversarial_ids.core.generator_runner import GeneratorRunner
from adversarial_ids.shared.json_io import load_json, save_json

DEFAULT_TRACE_ROOT = OUTPUTS_DIR / "experiments" / "traces"
DEFAULT_BENIGN_POOL = OUTPUTS_DIR / "experiments" / "benign_pool"
_BENIGN_ACTION_TMP = "config/actions/_experiment_benign.json"


class TraceError(RuntimeError):
    """Um trace (ou o benigno de uma seed) não pôde ser produzido."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_key(attack_key: str, config: dict[str, Any], seed: int, jar_sha: str) -> str:
    """Identidade de um trace: muda se qualquer um dos quatro mudar."""

    canonical = json.dumps(
        {"attack": attack_key, "config": config, "seed": int(seed), "jar": jar_sha},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


@dataclass
class TraceStore:
    """Gera e guarda traces por (ataque, config, seed), com benigno por seed."""

    root: Path = DEFAULT_TRACE_ROOT
    benign_pool: Path = DEFAULT_BENIGN_POOL
    runtime_dir: Path = GENERATOR_RUNTIME_DIR
    jar_path: Path = GENERATOR_JAR_PATH
    electrical_sources: Path = field(
        default_factory=lambda: ERENO_SOURCE_DIR / "electrical-sources" / "res10"
    )
    timeout_seconds: float = GENERATOR_TIMEOUT_SECONDS
    max_retries: int = GENERATOR_MAX_RETRIES

    @cached_property
    def jar_sha(self) -> str:
        if not self.jar_path.exists():
            raise TraceError(f"JAR do ERENO não encontrado: {self.jar_path}")
        return _sha256(self.jar_path)

    # ------------------------------------------------------------------ #
    # Benigno por seed                                                    #
    # ------------------------------------------------------------------ #
    def benign(self, seed: int) -> Path:
        """O tráfego normal da seed — gerado uma vez, reaproveitado sempre."""

        directory = self.benign_pool / f"seed_{int(seed)}"
        existing = self._benign_csv(directory)
        if existing is not None:
            return existing

        if not self.electrical_sources.is_dir():
            raise TraceError(
                f"Fontes elétricas não encontradas em {self.electrical_sources}; o "
                "CREATE_BENIGN precisa delas. Aponte ERENO_SOURCE_DIR para o ERENO-2.0."
            )
        action = load_json(self.runtime_dir / GENERATOR_BENIGN_ACTION_CONFIG_RELATIVE_PATH)
        action["randomSeed"] = int(seed)
        action["output"]["directory"] = directory.resolve().as_posix()
        action["electricalSources"] = {
            "directory": self.electrical_sources.as_posix(),
            "pattern": "SILVIO_*.out",
        }
        tmp = self.runtime_dir / _BENIGN_ACTION_TMP
        save_json(str(tmp), action)
        try:
            result = subprocess.run(
                [*GENERATOR_RUN_COMMAND[:-1], _BENIGN_ACTION_TMP],
                cwd=str(self.runtime_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        finally:
            tmp.unlink(missing_ok=True)
        produced = self._benign_csv(directory)
        if result.returncode != 0 or produced is None:
            tail = "\n".join(result.stderr.splitlines()[-5:])
            raise TraceError(f"CREATE_BENIGN falhou para a seed {seed}:\n{tail}")
        return produced

    @staticmethod
    def _benign_csv(directory: Path) -> Path | None:
        if not directory.is_dir():
            return None
        found = sorted(p for p in directory.glob("*.csv") if not p.name.endswith(".sv.csv"))
        return found[0] if found else None

    def _runtime_config_files(self) -> list[Path]:
        config_dir = self.runtime_dir / "config"
        return [
            path
            for sub in ("actions", "attacks")
            for path in (config_dir / sub).glob("*.json")
        ]

    def _snapshot_runtime_configs(self) -> dict[Path, bytes]:
        return {path: path.read_bytes() for path in self._runtime_config_files()}

    def _restore_runtime_configs(self, snapshot: dict[Path, bytes]) -> None:
        for path in self._runtime_config_files():
            if path not in snapshot:
                path.unlink(missing_ok=True)
        for path, content in snapshot.items():
            path.write_bytes(content)

    # ------------------------------------------------------------------ #
    # Trace de ataque                                                     #
    # ------------------------------------------------------------------ #
    def path_for(self, attack_key: str, config: dict[str, Any], seed: int) -> Path:
        key = config_key(attack_key, config, seed, self.jar_sha)
        return self.root / attack_key / f"{key}.csv.gz"

    def trace(self, attack_key: str, config: dict[str, Any], seed: int) -> Path:
        """O trace de ``config`` sob ``seed`` — do cache, ou gerado agora."""

        target = self.path_for(attack_key, config, seed)
        if target.exists():
            return target

        spec = get_attack_spec(attack_key)
        benign = self.benign(seed)
        work = self.root / "_work"
        work.mkdir(parents=True, exist_ok=True)
        runner = GeneratorRunner(
            runtime_dir=self.runtime_dir,
            output_dataset_path=GENERATOR_OUTPUT_DATASET_PATH,
            run_command=GENERATOR_RUN_COMMAND,
            suggested_config_path=str(work / "attack_config.json"),
            action_config_relative_path=GENERATOR_ACTION_CONFIG_RELATIVE_PATH,
            benign_action_config_relative_path=GENERATOR_BENIGN_ACTION_CONFIG_RELATIVE_PATH,
            segment_name=spec.segment_name,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            retry_backoff_seconds=GENERATOR_RETRY_BACKOFF_SECONDS,
            random_seed=int(seed),
            benign_data_path=benign,
        )
        # O GeneratorRunner reescreve o action config e o attack config
        # versionados do runtime a cada geração; restaurá-los aqui mantém a
        # árvore limpa durante um experimento de centenas de gerações.
        snapshot = self._snapshot_runtime_configs()
        try:
            produced = Path(runner.generate_dataset(config, iteration=0))
        except Exception as error:
            raise TraceError(f"Geração de {attack_key} (seed {seed}) falhou: {error}") from error
        finally:
            self._restore_runtime_configs(snapshot)

        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        with produced.open("rb") as source, gzip.open(tmp, "wb", compresslevel=6) as sink:
            shutil.copyfileobj(source, sink)
        tmp.replace(target)
        produced.unlink(missing_ok=True)
        save_json(
            str(target.with_suffix("").with_suffix(".json")),
            {
                "attack_key": attack_key,
                "seed": int(seed),
                "config": config,
                "jar_sha256": self.jar_sha,
                "benign_ref": benign.as_posix(),
                "benign_sha256": _sha256(benign),
            },
        )
        return target
