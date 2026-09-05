"""FeaturePreprocessor — pipeline fit/transform sem leakage (épico E6).

Extrai o preparo de features que hoje vive embutido e acoplado ao Random
Forest em ``core/ids_evaluator.py`` (``_fit_transform_features``/
``_transform_features``) para um componente independente, reaproveitável por
qualquer detector (RF, DT, SVM — épico E8) sob o mesmo protocolo.

Anti-leakage por construção
----------------------------
``fit`` só pode ver a partição de treino: decisão de coluna constante,
vocabulário categórico, valores de imputação e estatísticas do escalador são
todos ajustados exclusivamente sobre o dataframe passado a ``fit``. O chamador
(``IdsEvaluator``) é responsável por já ter feito o ``train_test_split`` antes
de chamar ``fit`` — este módulo não sabe (nem precisa saber) o que é treino ou
teste, só que tudo que recebe em ``fit`` é o que vira estatística ajustada.
``transform`` nunca recalcula nada a partir dos dados que recebe; só aplica o
que ``fit`` já decidiu. ``FeatureManifest.fitted_rows`` é a prova, no próprio
artefato, de que o ajuste viu só a partição de treino.

Os defaults reproduzem exatamente a semântica legada de
``core/ids_evaluator.py`` (``numeric_imputation="zero"``, sentinela
``"missing"`` + códigos ordinais de ``sorted(unique())`` para categóricas,
sem escala) — a correção de leakage não muda *o que* é calculado, só *sobre
quais linhas*. ``numeric_imputation="median"`` e ``scaler="standard"`` existem
para o E8 (SVM é sensível a escala); desligados por default.

Undersampling e seleção de features por mutual information (E7) não
pertencem aqui — consomem ``FeatureManifest.feature_columns`` como ponto de
partida, não fazem parte do fit/transform deste módulo. Ver
``core/feature_selector.py``, ``core/undersampler.py`` e
``docs/feature_selection.md``.
"""

from __future__ import annotations

import hashlib
from typing import Literal

import pandas as pd

from adversarial_ids.domain.feature_manifest import (
    DroppedColumn,
    FeatureManifest,
    ScalerStat,
)

# Mesma lista de sempre-descartar do avaliador legado (temporais/sequência,
# metadados/constantes do protocolo) — ver core/ids_evaluator.py para o
# histórico de cada coluna. Migrada para cá porque a decisão de descartar é
# do preprocessador, não do detector.
_ALWAYS_DROP: tuple[str, ...] = (
    # Temporais / sequência / derivados fortes
    "Time",
    "t",
    "GooseTimestamp",
    "receivedTimestamp",
    "timestampDiff",
    "tDiff",
    "timeFromLastChange",
    "delay",
    "SqNum",
    "StNum",
    "sqDiff",
    "stDiff",
    # Metadados/constantes comuns do protocolo
    "frameLen",
    "ethDst",
    "ethSrc",
    "ethType",
    "gooseTimeAllowedtoLive",
    "gooseAppid",
    "gooseLen",
    "TPID",
    "gocbRef",
    "datSet",
    "goID",
    "test",
    "confRev",
    "ndsCom",
    "numDatSetEntries",
    "APDUSize",
    "protocol",
    "gooseLengthDiff",
    "apduSizeDiff",
    "frameLengthDiff",
    "e2eLatency",
)

_MISSING_CATEGORY_SENTINEL = "missing"

# Teto defensivo de cardinalidade: no dataset real do ERENO todo objeto vira
# categórica só depois de sobreviver ao always_drop (e nenhuma sobrevive —
# ethDst/gocbRef/etc. estão todas na lista acima), então isto nunca dispara
# hoje. Existe para quando uma categórica de alta cardinalidade (um MAC, um
# ID, um timestamp-string) sobreviver: sem o teto, `categorical_codes` vira
# um mapa com uma entrada por valor distinto — um manifest de megabytes por
# execução em vez de alguns KB.
_MAX_CATEGORICAL_CARDINALITY = 1000


class PreprocessorError(ValueError):
    """Erro acionável: o preprocessador não pôde ajustar ou transformar os dados."""


class FeaturePreprocessor:
    """Fit/transform determinístico de features tabulares, sem leakage.

    ``fit`` decide, sobre o dataframe recebido, quais colunas descartar (e por
    quê), quais features são numéricas/categóricas, como imputar cada uma e —
    quando ``scaler="standard"`` — a média/desvio de cada numérica.
    ``transform`` aplica exatamente o que foi ajustado, tratando categoria
    nunca vista como ``unseen_category_code`` e coluna ausente no dataframe de
    entrada como ``missing_column_fill``. ``manifest()`` empacota o estado
    ajustado num ``FeatureManifest`` persistível; ``from_manifest`` reconstrói
    um preprocessador pronto para ``transform`` a partir desse artefato, sem
    precisar reajustar nem guardar pickle.
    """

    unseen_category_code = -1
    missing_column_fill = 0.0

    def __init__(
        self,
        *,
        always_drop: tuple[str, ...] = _ALWAYS_DROP,
        drop_cb_status: bool = False,
        numeric_imputation: Literal["zero", "median"] = "zero",
        scaler: Literal["none", "standard"] = "none",
    ) -> None:
        self.always_drop = always_drop
        self.drop_cb_status = drop_cb_status
        self.numeric_imputation = numeric_imputation
        self.scaler = scaler

        self._fitted = False
        self._label_column: str | None = None
        self._feature_columns: list[str] = []
        self._dropped_columns: list[DroppedColumn] = []
        self._numeric_features: list[str] = []
        self._categorical_features: list[str] = []
        self._numeric_fill_values: dict[str, float] = {}
        self._categorical_codes: dict[str, dict[str, int]] = {}
        self._scaler_stats: dict[str, ScalerStat] = {}
        self._fitted_rows = 0
        self._fitted_content_hash = ""

    # ------------------------------------------------------------------ #
    # fit / transform                                                     #
    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, *, label_column: str) -> "FeaturePreprocessor":
        """Ajusta o preprocessador sobre ``X`` — deve ser só a partição de treino.

        ``label_column`` não é usada em nenhum cálculo: é metadado ecoado no
        ``FeatureManifest`` para que o artefato diga, sozinho, qual coluna o
        preprocessador não viu (o rótulo é responsabilidade do chamador).
        """

        if X.empty:
            raise PreprocessorError("Não é possível ajustar sobre um dataframe vazio (0 linhas).")

        dropped: list[DroppedColumn] = []
        seen: set[str] = set()

        def _mark(column: str, reason: Literal["always_drop", "constant", "cb_status"]) -> None:
            if column not in seen:
                dropped.append(DroppedColumn(name=column, reason=reason))
                seen.add(column)

        # Precedência de motivo quando uma coluna se qualifica para mais de um
        # (comum no dataset real: 21 das 33 colunas descartadas são
        # simultaneamente sempre-descartar E constantes no treino):
        # always_drop > cb_status > constant, nessa ordem — cada laço abaixo
        # só marca colunas que os laços anteriores ainda não marcaram.
        for column in X.columns:
            if column in self.always_drop:
                _mark(column, "always_drop")

        if self.drop_cb_status:
            for column in ("cbStatus", "cbStatusDiff"):
                if column in X.columns:
                    _mark(column, "cb_status")

        for column in X.columns:
            if column in seen:
                continue
            # nunique(dropna=False) <= 1 cobre tanto "um único valor" quanto
            # "só NaN" (NaN conta como um valor distinto) — não há reason
            # "all_nan" separado porque nenhuma coluna passa por aqui sem
            # antes cair neste mesmo check.
            if X[column].nunique(dropna=False) <= 1:
                _mark(column, "constant")

        feature_columns = [column for column in X.columns if column not in seen]

        if not feature_columns:
            raise PreprocessorError(
                "Nenhuma feature sobrou após remover colunas sempre-descartadas e constantes."
            )

        X_features = X[feature_columns]

        numeric_features: list[str] = []
        categorical_features: list[str] = []
        numeric_fill_values: dict[str, float] = {}
        categorical_codes: dict[str, dict[str, int]] = {}

        for column in feature_columns:
            if pd.api.types.is_numeric_dtype(X_features[column]):
                numeric_features.append(column)
                if self.numeric_imputation == "median":
                    fill_value = float(pd.to_numeric(X_features[column], errors="coerce").median())
                else:
                    fill_value = self.missing_column_fill
                numeric_fill_values[column] = fill_value
            else:
                categorical_features.append(column)
                normalized = X_features[column].fillna(_MISSING_CATEGORY_SENTINEL).astype(str)
                uniques = sorted(normalized.unique())
                if len(uniques) > _MAX_CATEGORICAL_CARDINALITY:
                    raise PreprocessorError(
                        f"Coluna categórica {column!r} tem {len(uniques)} valores distintos no "
                        f"treino, acima do teto de {_MAX_CATEGORICAL_CARDINALITY} — provavelmente "
                        "não é uma categórica de verdade (ex.: um ID ou timestamp em texto)."
                    )
                categorical_codes[column] = {value: index for index, value in enumerate(uniques)}

        scaler_stats: dict[str, ScalerStat] = {}
        if self.scaler == "standard":
            for column in numeric_features:
                series = pd.to_numeric(X_features[column], errors="coerce").fillna(
                    numeric_fill_values[column]
                )
                mean = float(series.mean())
                std = float(series.std())
                if not std or pd.isna(std):
                    # Coluna quase-constante no treino: desvio 1 em vez de 0
                    # evita divisão por zero e não amplifica ruído no
                    # transform (o valor centrado já carrega toda a variação).
                    std = 1.0
                scaler_stats[column] = ScalerStat(mean=mean, std=std)

        self._label_column = label_column
        self._feature_columns = feature_columns
        self._dropped_columns = dropped
        self._numeric_features = numeric_features
        self._categorical_features = categorical_features
        self._numeric_fill_values = numeric_fill_values
        self._categorical_codes = categorical_codes
        self._scaler_stats = scaler_stats
        self._fitted_rows = len(X)
        self._fitted_content_hash = self._hash_frame(X)
        self._fitted = True

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica o estado ajustado por ``fit`` (ou reconstruído por ``from_manifest``)."""

        if not self._fitted:
            raise PreprocessorError("transform() chamado antes de fit(): nenhum estado ajustado.")

        X = X.copy()
        dropped_names = {column.name for column in self._dropped_columns}
        X = X.drop(columns=[column for column in dropped_names if column in X.columns], errors="ignore")

        for column in self._feature_columns:
            if column not in X.columns:
                X[column] = self.missing_column_fill

        X = X[self._feature_columns].copy()

        for column in self._numeric_features:
            X[column] = pd.to_numeric(X[column], errors="coerce").fillna(
                self._numeric_fill_values[column]
            )
            if self.scaler == "standard":
                stat = self._scaler_stats[column]
                X[column] = (X[column] - stat.mean) / stat.std

        for column in self._categorical_features:
            mapping = self._categorical_codes[column]
            X[column] = (
                X[column]
                .fillna(_MISSING_CATEGORY_SENTINEL)
                .astype(str)
                .map(mapping)
                .fillna(self.unseen_category_code)
                .astype(int)
            )

        return X

    def fit_transform(self, X: pd.DataFrame, *, label_column: str) -> pd.DataFrame:
        self.fit(X, label_column=label_column)
        return self.transform(X)

    # ------------------------------------------------------------------ #
    # Manifest — persistência sem pickle                                  #
    # ------------------------------------------------------------------ #
    def manifest(self) -> FeatureManifest:
        if not self._fitted:
            raise PreprocessorError("manifest() chamado antes de fit(): nenhum estado ajustado.")

        assert self._label_column is not None  # garantido por fit()

        return FeatureManifest(
            label_column=self._label_column,
            feature_columns=tuple(self._feature_columns),
            dropped_columns=tuple(self._dropped_columns),
            numeric_features=tuple(self._numeric_features),
            categorical_features=tuple(self._categorical_features),
            numeric_imputation=self.numeric_imputation,
            numeric_fill_values=dict(self._numeric_fill_values),
            categorical_codes={k: dict(v) for k, v in self._categorical_codes.items()},
            scaler=self.scaler,
            scaler_stats=dict(self._scaler_stats),
            fitted_rows=self._fitted_rows,
            fitted_content_hash=self._fitted_content_hash,
        )

    @classmethod
    def from_manifest(cls, manifest: FeatureManifest) -> "FeaturePreprocessor":
        """Reconstrói um preprocessador pronto para ``transform`` a partir de um manifest.

        Não reajusta nada — só recarrega o estado que ``fit`` já tinha
        decidido. ``fit``/``fit_transform`` continuam indisponíveis de forma
        útil numa instância assim (chamá-los reajusta do zero, descartando o
        estado importado), o que é o comportamento certo: um preprocessador
        vindo de manifest existe para reproduzir um transform já congelado,
        não para virar um novo fit.
        """

        preprocessor = cls(
            drop_cb_status=any(column.reason == "cb_status" for column in manifest.dropped_columns),
            numeric_imputation=manifest.numeric_imputation,
            scaler=manifest.scaler,
        )
        preprocessor._label_column = manifest.label_column
        preprocessor._feature_columns = list(manifest.feature_columns)
        preprocessor._dropped_columns = list(manifest.dropped_columns)
        preprocessor._numeric_features = list(manifest.numeric_features)
        preprocessor._categorical_features = list(manifest.categorical_features)
        preprocessor._numeric_fill_values = dict(manifest.numeric_fill_values)
        preprocessor._categorical_codes = {k: dict(v) for k, v in manifest.categorical_codes.items()}
        preprocessor._scaler_stats = dict(manifest.scaler_stats)
        preprocessor._fitted_rows = manifest.fitted_rows
        preprocessor._fitted_content_hash = manifest.fitted_content_hash
        preprocessor._fitted = True

        return preprocessor

    # ------------------------------------------------------------------ #
    # Introspecção pública (paridade com os atributos que                #
    # core/ids_evaluator.py expunha antes do refactor)                    #
    # ------------------------------------------------------------------ #
    @property
    def feature_columns(self) -> list[str]:
        return list(self._feature_columns)

    @property
    def removed_columns(self) -> list[str]:
        return sorted(column.name for column in self._dropped_columns)

    @staticmethod
    def _hash_frame(X: pd.DataFrame) -> str:
        # `lineterminator="\n"` fixo: o padrão do pandas usa `os.linesep` ao
        # gravar em arquivo, o que faria o mesmo dataframe produzir hashes
        # diferentes no Windows (CRLF) e no Linux/CI (LF).
        content = X.to_csv(index=False, lineterminator="\n").encode("utf-8")
        return hashlib.sha256(content).hexdigest()
