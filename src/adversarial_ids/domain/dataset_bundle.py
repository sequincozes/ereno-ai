"""DatasetBundle — dataset preparado a partir de um trace do ERENO.

Contrato congelado (ação 72h #2). Representa a saída da etapa PREPROCESS
(schema, classes, contagens, hash e linhagem) — o fit/transform sem leakage
(imputação, encoding, escala) pertence ao ``FeatureManifest`` do épico E6
(``domain/feature_manifest.py``, ``core/preprocessor.py``), que consome este
``DatasetBundle`` já aprovado; este contrato só garante a consistência
interna do artefato: as classes declaradas e as contagens cobrem o mesmo
conjunto de rótulos, e nenhuma classe fica vazia.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetBundle(BaseModel):
    """Dataset preparado, rastreável até o trace e a execução que o gerou."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    trace_path: str = Field(min_length=1)
    columns: tuple[str, ...] = Field(min_length=1)
    classes: tuple[str, ...] = Field(min_length=1)
    class_counts: dict[str, int]
    content_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    lineage_run_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _classes_and_counts_agree(self) -> "DatasetBundle":
        if set(self.classes) != set(self.class_counts):
            raise ValueError(
                "classes e class_counts devem cobrir exatamente o mesmo conjunto de rótulos."
            )
        if any(count <= 0 for count in self.class_counts.values()):
            raise ValueError("class_counts não pode conter contagens não positivas.")
        return self
