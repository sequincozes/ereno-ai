import csv
import shutil
import subprocess
from pathlib import Path
from typing import Any

from adversarial_ids.shared.json_io import load_json, save_json


class GeneratorRunner:
    """Produz o dataset de cada iteração.

    Dois modos:

    - ``jar``    — executa o gerador ERENO (Java) sobre o attack_config.
    - ``cached`` — **não usa Java**: devolve um dataset benigno+ataque já
      versionado (``data/baseline_dataset.csv``). É o que permite M1/M2/M4 e o
      CI rodarem o loop ponta a ponta sem o JAR. Em modo cacheado o dataset é o
      mesmo em toda iteração — o objetivo é destravar o fluxo (agentes, memória,
      dashboard), não reproduzir a variação física por config.

    Seleção do ataque (modo ``jar``)
    --------------------------------
    Quando ``segment_name`` é informado, o runner passa a ser **multi-ataque**:
    a cada geração ele reescreve o ``action`` config habilitando **apenas** aquele
    segmento (e desabilitando os demais) e grava o attack config no caminho que o
    segmento aponta. O rótulo da classe sai correto porque o JAR o deriva do
    prefixo ``ucXX`` do nome do segmento. Sem ``segment_name`` mantém-se o
    comportamento histórico (uc03 já habilitado no action config).
    """

    def __init__(
        self,
        runtime_dir: Path,
        output_dataset_path: Path,
        run_command: list[str],
        suggested_config_path: str,
        attack_config_relative_path: str | None = None,
        action_config_relative_path: str | None = None,
        benign_action_config_relative_path: str | None = None,
        benign_seed_path: Path | str | None = None,
        segment_name: str | None = None,
        cached_dataset_path: Path | str | None = None,
    ) -> None:
        self.runtime_dir = Path(runtime_dir)
        self.output_dataset_path = Path(output_dataset_path)
        self.run_command = run_command
        self.suggested_config_path = Path(suggested_config_path)
        self.attack_config_relative_path = attack_config_relative_path
        self.action_config_relative_path = action_config_relative_path
        self.benign_action_config_relative_path = benign_action_config_relative_path
        self.benign_seed_path = (
            Path(benign_seed_path) if benign_seed_path is not None else None
        )
        self.segment_name = segment_name
        self.cached_dataset_path = (
            Path(cached_dataset_path) if cached_dataset_path is not None else None
        )

    @property
    def is_cached(self) -> bool:
        return self.cached_dataset_path is not None

    def generate_dataset(
        self,
        attack_config: dict[str, Any],
        iteration: int,
    ) -> str:
        self.suggested_config_path.parent.mkdir(parents=True, exist_ok=True)

        # Persistir os configs sempre — o resto do pipeline (diffs, logs,
        # histórico) continua idêntico nos dois modos.
        save_json(str(self.suggested_config_path), attack_config)

        iteration_dataset_path = (
            self.suggested_config_path.parent
            / f"dataset_iteration_{iteration}.csv"
        )

        if self.is_cached:
            return self._generate_from_cache(iteration_dataset_path)

        return self._generate_from_jar(attack_config, iteration_dataset_path)

    def _generate_from_cache(self, iteration_dataset_path: Path) -> str:
        if self.cached_dataset_path is None or not self.cached_dataset_path.exists():
            raise FileNotFoundError(
                "Modo cacheado ativo, mas o dataset semente não foi encontrado: "
                f"{self.cached_dataset_path}. "
                "Gere/forneça data/baseline_dataset.csv ou use GENERATOR_MODE=jar."
            )

        shutil.copyfile(self.cached_dataset_path, iteration_dataset_path)
        print(f"[GEN:cached] Dataset servido do cache: {iteration_dataset_path}")

        return str(iteration_dataset_path)

    # ------------------------------------------------------------------ #
    # Resolução do caminho do attack config e do action config           #
    # ------------------------------------------------------------------ #
    def _action_config_path(self) -> Path:
        if self.action_config_relative_path is None:
            raise RuntimeError(
                "action_config_relative_path não configurado, mas segment_name foi "
                "informado — não é possível selecionar o segmento do ataque."
            )
        return self.runtime_dir / self.action_config_relative_path

    def _select_segment_and_resolve_attack_path(self) -> Path:
        """Habilita só o segmento escolhido no action config e devolve o caminho
        (absoluto) do attack config que aquele segmento espera."""

        action_path = self._action_config_path()
        action = load_json(action_path)

        segments = action.get("attackSegments", [])
        target = None
        for segment in segments:
            is_target = segment.get("name") == self.segment_name
            segment["enabled"] = is_target
            if is_target:
                target = segment

        if target is None:
            available = ", ".join(s.get("name", "?") for s in segments)
            raise ValueError(
                f"Segmento {self.segment_name!r} não encontrado no action config "
                f"{action_path}. Segmentos disponíveis: {available}."
            )

        attack_rel = target.get("attackConfig")
        if not attack_rel:
            raise ValueError(
                f"Segmento {self.segment_name!r} não define 'attackConfig' no action config."
            )

        save_json(str(action_path), action)
        return self.runtime_dir / attack_rel

    def _resolve_attack_config_path(self) -> Path:
        if self.segment_name is not None:
            return self._select_segment_and_resolve_attack_path()
        if self.attack_config_relative_path is None:
            raise RuntimeError(
                "Nem segment_name nem attack_config_relative_path foram configurados; "
                "não há onde gravar o attack config."
            )
        return self.runtime_dir / self.attack_config_relative_path

    def _generate_from_jar(
        self,
        attack_config: dict[str, Any],
        iteration_dataset_path: Path,
    ) -> str:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.output_dataset_path.parent.mkdir(parents=True, exist_ok=True)

        attack_config_path = self._resolve_attack_config_path()
        attack_config_path.parent.mkdir(parents=True, exist_ok=True)

        save_json(str(attack_config_path), attack_config)

        self._ensure_benign_dataset()

        result = subprocess.run(
            self.run_command,
            cwd=str(self.runtime_dir),
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "Synthetic generator execution failed.\n"
                f"Command: {' '.join(self.run_command)}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

        if not self.output_dataset_path.exists():
            raise FileNotFoundError(
                f"Expected generated dataset was not found: {self.output_dataset_path}"
            )

        shutil.copyfile(self.output_dataset_path, iteration_dataset_path)

        return str(iteration_dataset_path)

    def _ensure_benign_dataset(self) -> None:
        """Gera e conecta o dataset benigno quando o action config aponta
        para um arquivo ainda inexistente.

        Primeiro tenta extrair registros normais do seed local. Sem um seed,
        usa ``CREATE_BENIGN``; como essa ação inclui um timestamp no nome do
        CSV, atualiza ``benignDataPath`` com o arquivo efetivamente produzido.
        """
        action_path = self._action_config_path()
        action = load_json(action_path)
        benign_rel = action.get("input", {}).get("benignDataPath")
        if not benign_rel:
            raise ValueError(
                f"O action config {action_path} não define input.benignDataPath."
            )

        benign_path = self.runtime_dir / benign_rel
        if benign_path.exists():
            return

        benign_dir = benign_path.parent
        benign_dir.mkdir(parents=True, exist_ok=True)
        print(
            "[GEN:BENIGN] AUSENTE | Dataset benigno não encontrado; "
            "a preparação automática foi iniciada: "
            f"{benign_path}"
        )

        if self.benign_seed_path is not None and self.benign_seed_path.exists():
            normal_rows = self._extract_benign_seed(
                self.benign_seed_path,
                benign_path,
            )
            print(
                "[GEN:BENIGN] CRIADO | Dataset benigno extraído do seed "
                f"({normal_rows} registros): {benign_path}"
            )
            return

        if self.benign_action_config_relative_path is None:
            raise FileNotFoundError(
                f"Dataset benigno ausente: {benign_path}. Não há seed local "
                "nem configuração CREATE_BENIGN para recuperá-lo."
            )

        benign_config_path = (
            self.runtime_dir / self.benign_action_config_relative_path
        )
        if not benign_config_path.exists():
            raise FileNotFoundError(
                f"Configuração para geração benigna não encontrada: "
                f"{benign_config_path}"
            )

        existing = set(benign_dir.glob("*.csv"))

        benign_command = [
            *self.run_command[:-1],
            self.benign_action_config_relative_path,
        ]
        result = subprocess.run(
            benign_command,
            cwd=str(self.runtime_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "A geração automática do dataset benigno falhou.\n"
                f"Command: {' '.join(benign_command)}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

        generated = [
            path
            for path in benign_dir.glob("*.csv")
            if path not in existing and not path.name.endswith(".sv.csv")
        ]
        if not generated:
            raise FileNotFoundError(
                "A ação CREATE_BENIGN terminou sem produzir um novo CSV em "
                f"{benign_dir}."
            )

        generated_path = max(generated, key=lambda path: path.stat().st_mtime_ns)
        action["input"]["benignDataPath"] = generated_path.relative_to(
            self.runtime_dir
        ).as_posix()
        save_json(str(action_path), action)
        print(f"[GEN:BENIGN] CRIADO | Dataset benigno criado: {generated_path}")

    @staticmethod
    def _extract_benign_seed(seed_path: Path, destination: Path) -> int:
        """Extrai do dataset combinado apenas os registros ``normal``."""
        with seed_path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.reader(source)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ValueError(f"Dataset seed vazio: {seed_path}") from exc

            normalized_header = [column.strip() for column in header]
            try:
                class_index = normalized_header.index("class")
            except ValueError as exc:
                raise ValueError(
                    f"Dataset seed não possui a coluna 'class': {seed_path}"
                ) from exc

            count = 0
            with destination.open("w", encoding="utf-8", newline="") as target:
                writer = csv.writer(target, lineterminator="\n")
                writer.writerow(header)
                for row in reader:
                    if len(row) <= class_index:
                        continue
                    if row[class_index].strip().lower() != "normal":
                        continue
                    writer.writerow(row)
                    count += 1

        if count == 0:
            destination.unlink(missing_ok=True)
            raise ValueError(
                f"Dataset seed não contém registros com classe 'normal': {seed_path}"
            )
        return count
