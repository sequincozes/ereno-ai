import shutil
import subprocess
from pathlib import Path
from typing import Any

from tools.json_loader import save_json


class ErenoRunner:
    def __init__(
        self,
        ereno_project_path: str,
        attack_config_relative_path: str,
        output_dataset_path: str,
        run_command: list[str],
        suggested_config_path: str,
    ) -> None:
        self.ereno_project_path = Path(ereno_project_path)
        self.attack_config_path = self.ereno_project_path / attack_config_relative_path
        self.output_dataset_path = Path(output_dataset_path)
        self.run_command = run_command
        self.suggested_config_path = Path(suggested_config_path)

    def generate_dataset(self, attack_config: dict[str, Any], iteration: int) -> str:
        """
        Salva o JSON dentro do projeto Java do ERENO e executa o comando do ERENO.

        Nesta primeira versão, o agente ainda usa o JSON atual.
        Depois podemos conectar a resposta da LLM para gerar um novo JSON automaticamente.
        """

        print(f"[ERENO] Salvando configuração sugerida em: {self.suggested_config_path}")
        save_json(self.suggested_config_path, attack_config)

        print(f"[ERENO] Salvando configuração no projeto Java: {self.attack_config_path}")
        save_json(self.attack_config_path, attack_config)

        print("[ERENO] Executando projeto Java...")
        print(f"[ERENO] Diretório: {self.ereno_project_path}")
        print(f"[ERENO] Comando: {' '.join(self.run_command)}")

        result = subprocess.run(
            self.run_command,
            cwd=self.ereno_project_path,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            print("[ERENO] Erro na execução.")
            print(result.stderr)
            raise RuntimeError("Falha ao executar o ERENO.")

        print("[ERENO] Execução finalizada.")
        print(result.stdout)

        if not self.output_dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset esperado não encontrado em: {self.output_dataset_path}"
            )

        iteration_dataset_path = Path("outputs") / f"dataset_iteration_{iteration}.csv"
        iteration_dataset_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy(self.output_dataset_path, iteration_dataset_path)

        print(f"[ERENO] Dataset copiado para: {iteration_dataset_path}")

        return str(iteration_dataset_path)