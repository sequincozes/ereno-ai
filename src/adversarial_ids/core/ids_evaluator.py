from typing import Any, Mapping

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from adversarial_ids.config.settings import (
    DETECTOR_MODE,
    DETECTOR_SCALER,
    FEATURE_SELECTION_MIN_SCORE,
    FEATURE_SELECTION_MODE,
    FEATURE_SELECTION_TOP_K,
    PREPROCESSOR_MODE,
    PROTOCOL_FEATURES_MODE,
    UNDERSAMPLING_MODE,
)
from adversarial_ids.core.detectors import Detector, recommended_scaler
from adversarial_ids.core.feature_selector import FeatureSelector
from adversarial_ids.core.preprocessor import FeaturePreprocessor, always_drop_for
from adversarial_ids.core.undersampler import RandomUndersampler
from adversarial_ids.domain.detector_manifest import DetectorManifest
from adversarial_ids.domain.feature_manifest import FeatureManifest
from adversarial_ids.domain.selection_manifest import SelectionManifest


class IdsEvaluator:
    """
    Avaliador do IDS.

    Treina o detector apenas no baseline e testa cada variante
    sem retreinar o modelo.

    Detector plugável (épico E8): o modelo não é mais um
    ``RandomForestClassifier`` embutido — é um ``core.detectors.Detector``
    resolvido por chave (``"random_forest"`` default, mais ``"decision_tree"``,
    ``"svm_linear"`` e ``"svm_rbf"``). Todos entram no **mesmo ponto** do
    pipeline, depois do mesmo split e do mesmo preparo, o que é o que torna a
    comparação entre eles honesta (etapa D36-46: "mesmo split/protocolo para
    todos os detectores"). O default reproduz o comportamento anterior ao E8 —
    mesmos hiperparâmetros, mesma ordem de fit, mesmas importâncias Gini.
    Ver ``docs/detectors.md``.

    Ressalva de comparabilidade: "mesmo preparo" é literal enquanto
    ``feature_selection="none"`` (o default). Com ``"mutual_info"`` a seleção
    é ajustada sobre o X **já escalado**, e ``mutual_info_classif`` não é
    estritamente invariante a escala — dois detectores com recomendações de
    escala diferentes podem, em princípio, receber conjuntos de features
    diferentes. Numa comparação com seleção ligada, passe ``scaler=``
    explicitamente para forçar os dois lados à mesma escala.

    Preprocessamento de features (épico E6): o preparo de features (colunas
    descartadas, imputação, encoding categórico) não vive mais aqui — foi
    extraído para ``core.preprocessor.FeaturePreprocessor``, reaproveitável
    por qualquer detector (E8) sob o mesmo protocolo. O que muda entre os dois
    ``preprocessor_mode`` é só **quando** o ``fit`` acontece em relação ao
    ``train_test_split``, não a lógica de preprocessamento em si — as duas
    variantes chamam o mesmo ``FeaturePreprocessor``:

    - ``"modular"`` (default, ``config.settings.PREPROCESSOR_MODE``): o split
      acontece sobre o X **cru**, e o ``fit`` só depois, exclusivamente na
      partição de treino — nenhuma decisão de coluna constante ou vocabulário
      categórico vê as linhas de teste. Sem leakage.
    - ``"legacy"``: ajusta o ``FeaturePreprocessor`` sobre o dataset **inteiro**
      antes do split — reproduz o comportamento anterior ao E6, só para
      comparar métricas antes/depois da correção (ver ``docs/preprocessing.md``).

    Undersampling e seleção de features (épico E7, ``core/undersampler.py`` e
    ``core/feature_selector.py``): rodam **depois** do preprocessador, só
    sobre a partição de treino já transformada — nesta ordem: seleção de
    features primeiro (mutual information é mais estável com mais linhas, e
    a classe minoritária de ataque ainda não perdeu peso), undersampling só
    depois, no espaço já reduzido. Ambos default ``"none"`` — comportamento
    idêntico a antes do E7 até alguém optar explicitamente por uma
    estratégia (ver ``docs/feature_selection.md``).

    Escala (E6 × E8): ``scaler=None`` (default) delega ao detector — os SVMs
    pedem ``"standard"``, as árvores ``"none"``. Passar ``scaler="none"`` ou
    ``"standard"`` explicitamente força os dois lados a usarem a mesma escala,
    que é o que uma ablação controlada entre detectores precisa; o
    ``DetectorManifest`` registra a recomendação e o valor aplicado lado a
    lado, então um SVM rodado sem escala fica visível no artefato.

    O `LabelEncoder` do rótulo fica de fora do preprocessador de propósito: um
    espaço de rótulos não é uma estatística ajustada sobre features, é a
    definição das classes que o modelo aprende a prever — ajustá-lo no
    dataset inteiro (treino+teste) não vaza informação de features para o
    treino, só fixa o vocabulário de classes que já é conhecido de antemão.
    """

    _VALID_PREPROCESSOR_MODES = ("modular", "legacy")
    _VALID_SCALERS = ("none", "standard")

    def __init__(
        self,
        test_size: float = 0.3,
        random_state: int = 42,
        n_estimators: int | None = None,
        drop_cb_status: bool = False,
        target_attack_label: str | None = None,
        preprocessor_mode: str = PREPROCESSOR_MODE,
        protocol_features: str = PROTOCOL_FEATURES_MODE,
        feature_selection: str = FEATURE_SELECTION_MODE,
        feature_selection_top_k: int | None = FEATURE_SELECTION_TOP_K,
        feature_selection_min_score: float | None = FEATURE_SELECTION_MIN_SCORE,
        undersampling: str = UNDERSAMPLING_MODE,
        detector: str = DETECTOR_MODE,
        detector_hyperparameters: Mapping[str, Any] | None = None,
        scaler: str | None = DETECTOR_SCALER,
    ) -> None:
        if preprocessor_mode not in self._VALID_PREPROCESSOR_MODES:
            raise ValueError(
                f"preprocessor_mode inválido: {preprocessor_mode!r}. "
                f"Use um de {self._VALID_PREPROCESSOR_MODES}."
            )
        if scaler is not None and scaler not in self._VALID_SCALERS:
            raise ValueError(
                f"scaler inválido: {scaler!r}. Use um de {self._VALID_SCALERS} "
                "ou None para delegar ao detector."
            )

        self.test_size = test_size
        self.random_state = random_state
        # ``None`` = "não informado". A sentinela existe para distinguir isso
        # de "informado com o valor default", que é o que permite recusar
        # `n_estimators` num detector que não é Random Forest (ver abaixo)
        # em vez de descartá-lo em silêncio. ``self.n_estimators`` é
        # reatribuído depois da resolução, para nunca contradizer o modelo.
        self.drop_cb_status = drop_cb_status
        # Validado já aqui, e não só na construção do preprocessador lá em
        # train_baseline: um modo inválido precisa falhar quando o avaliador é
        # criado, não no meio de um treino.
        always_drop_for(protocol_features)
        self.protocol_features = protocol_features
        # Rótulo da classe de ataque esperado (ex.: "random_replay", "grayhole").
        # Quando definido, ancora a detecção da classe de ataque em vez de depender
        # de heurística por substring — essencial para ataques cujo rótulo não
        # contém "attack"/"masquerade".
        self.target_attack_label = target_attack_label
        self.preprocessor_mode = preprocessor_mode
        # Épico E7 — ver docstring da classe para a ordem seleção→undersampling.
        self.feature_selection = feature_selection
        self.feature_selection_top_k = feature_selection_top_k
        self.feature_selection_min_score = feature_selection_min_score
        self.undersampling = undersampling

        # Épico E8 — detector plugável. ``n_estimators`` é um hiperparâmetro
        # exclusivo do Random Forest: continua aceito na assinatura (é anterior
        # ao E8 e há chamadores usando), mas informá-lo para outro detector, ou
        # informá-lo duas vezes, é erro em vez de silêncio — o mesmo rigor com
        # que ``Detector`` recusa hiperparâmetro não suportado.
        overrides: dict[str, Any] = dict(detector_hyperparameters or {})
        if n_estimators is not None:
            if detector != "random_forest":
                raise ValueError(
                    f"n_estimators={n_estimators!r} só se aplica a "
                    f"detector='random_forest' (recebido {detector!r}). Use "
                    "detector_hyperparameters para os demais detectores."
                )
            if "n_estimators" in overrides:
                raise ValueError(
                    "n_estimators informado duas vezes (argumento direto="
                    f"{n_estimators!r} e detector_hyperparameters="
                    f"{overrides['n_estimators']!r}) — escolha uma das duas fontes."
                )
            overrides["n_estimators"] = n_estimators

        self.detector = Detector(
            detector, random_state=self.random_state, hyperparameters=overrides
        )
        # Espelha o que o estimador realmente recebeu (``None`` para detectores
        # que não têm o conceito) — nunca um valor que o modelo não usa.
        self.n_estimators = self.detector.hyperparameters.get("n_estimators")
        # ``None`` = delegar ao detector; um valor explícito vence a
        # recomendação (ver docstring da classe).
        self.scaler = scaler if scaler is not None else recommended_scaler(detector)

        self.model: Any | None = None
        self.label_column: str | None = None
        self.preprocessor: FeaturePreprocessor | None = None
        self.feature_selector: FeatureSelector | None = None
        self.undersampler: RandomUndersampler | None = None
        self._class_counts_before_undersampling: dict[str, int] | None = None
        self._class_counts_after_undersampling: dict[str, int] | None = None
        self._fitted_rows_before_undersampling: int | None = None
        self._fitted_rows_after_undersampling: int | None = None

        self.label_encoder: LabelEncoder | None = None
        self.class_mapping: dict[int, str] = {}
        self.attack_label: int | None = None

    @property
    def feature_columns(self) -> list[str] | None:
        """Features que o modelo realmente treinou — depois da seleção (E7).

        ``feature_selector`` está sempre ajustado depois de ``train_baseline``
        (mesmo com ``strategy="none"``, onde ``selected_features`` é igual às
        candidatas do preprocessador) — por isso este é o espaço certo para
        zipar com ``model.feature_importances_``, nunca
        ``self.preprocessor.feature_columns`` diretamente.
        """
        if self.feature_selector is not None:
            return self.feature_selector.selected_features
        return self.preprocessor.feature_columns if self.preprocessor is not None else None

    @property
    def removed_columns(self) -> list[str]:
        return self.preprocessor.removed_columns if self.preprocessor is not None else []

    @property
    def feature_manifest(self) -> FeatureManifest | None:
        """Manifest do ``FeaturePreprocessor`` ajustado (E6).

        ``None`` antes de ``train_baseline`` — não há estado ajustado para
        empacotar ainda.
        """
        return self.preprocessor.manifest() if self.preprocessor is not None else None

    @property
    def selection_manifest(self) -> SelectionManifest | None:
        """Manifest de seleção de features + undersampling ajustado (E7).

        ``None`` antes de ``train_baseline``, mesma semântica de
        ``feature_manifest``. Presente mesmo quando as duas estratégias são
        ``"none"`` — o manifest também documenta que nada foi filtrado/
        reamostrado, não só quando algo foi.
        """
        if (
            self.feature_selector is None
            or self._fitted_rows_before_undersampling is None
            or self._fitted_rows_after_undersampling is None
        ):
            return None

        return SelectionManifest(
            feature_selection=self.feature_selection,
            feature_selection_top_k=self.feature_selection_top_k,
            feature_selection_min_score=self.feature_selection_min_score,
            candidate_features=tuple(self.feature_selector.candidate_features),
            selected_features=tuple(self.feature_selector.selected_features),
            feature_scores=self.feature_selector.scores,
            undersampling=self.undersampling,
            undersampling_random_state=self.random_state,
            class_counts_before=self._class_counts_before_undersampling or {},
            class_counts_after=self._class_counts_after_undersampling or {},
            fitted_rows_before_undersampling=self._fitted_rows_before_undersampling,
            fitted_rows_after_undersampling=self._fitted_rows_after_undersampling,
        )

    @property
    def detector_manifest(self) -> DetectorManifest | None:
        """Manifest do detector treinado (E8).

        ``None`` antes de ``train_baseline``, mesma semântica de
        ``feature_manifest``/``selection_manifest``. Carrega a escala
        efetivamente aplicada pelo preprocessador, não só a que o detector
        recomenda — é o que fecha a prova de "mesmo protocolo" entre dois
        detectores comparados.
        """
        if not self.detector.is_fitted:
            return None
        return self.detector.manifest(resolved_scaler=self.scaler)

    def train_baseline(self, dataset_path: str) -> dict[str, Any]:
        print(f"[IDS] Treinando modelo baseline com: {dataset_path}")

        df = pd.read_csv(dataset_path)

        if df.empty:
            raise ValueError("Dataset baseline vazio.")

        self.label_column = self._find_label_column(df)
        print(f"[IDS] Coluna de classe detectada: {self.label_column}")

        X = df.drop(columns=[self.label_column])
        y = df[self.label_column]
        y_encoded = self._fit_transform_labels(y)

        # ``scaler`` já vem resolvido do __init__ (recomendação do detector ou
        # override explícito do chamador) — é o único ponto do E8 que toca o
        # preparo de features do E6.
        self.preprocessor = FeaturePreprocessor(
            protocol_features=self.protocol_features,
            drop_cb_status=self.drop_cb_status,
            scaler=self.scaler,  # type: ignore[arg-type]  # validado no __init__
        )

        if self.preprocessor_mode == "legacy":
            # Ajusta sobre o dataset INTEIRO antes do split — reproduz de
            # propósito o leakage que o modo "modular" (default) elimina; só
            # para comparação (ver docstring da classe).
            X_prepared = self.preprocessor.fit_transform(X, label_column=self.label_column)
            X_train, X_test, y_train, y_test = train_test_split(
                X_prepared,
                y_encoded,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y_encoded if len(set(y_encoded)) > 1 else None,
            )
        else:
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X,
                y_encoded,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y_encoded if len(set(y_encoded)) > 1 else None,
            )
            X_train = self.preprocessor.fit_transform(X_train_raw, label_column=self.label_column)
            X_test = self.preprocessor.transform(X_test_raw)

        # Épico E7 — seleção de features primeiro (ajustada sobre TODO o
        # treino, antes de qualquer undersampling), undersampling depois, só
        # no espaço já reduzido. Nunca sobre X_test/y_test: ver docstring da
        # classe e de core/undersampler.py.
        self.feature_selector = FeatureSelector(
            strategy=self.feature_selection,
            top_k=self.feature_selection_top_k,
            min_score=self.feature_selection_min_score,
            random_state=self.random_state,
        )
        X_train = self.feature_selector.fit_transform(X_train, y_train)
        X_test = self.feature_selector.transform(X_test)

        self.undersampler = RandomUndersampler(
            strategy=self.undersampling, random_state=self.random_state
        )
        self._fitted_rows_before_undersampling = len(X_train)
        self._class_counts_before_undersampling = {
            self.class_mapping.get(int(label), str(label)): count
            for label, count in RandomUndersampler.class_counts(y_train).items()
        }
        X_train, y_train = self.undersampler.fit_resample(X_train, y_train)
        self._fitted_rows_after_undersampling = len(X_train)
        self._class_counts_after_undersampling = {
            self.class_mapping.get(int(label), str(label)): count
            for label, count in RandomUndersampler.class_counts(y_train).items()
        }

        print(f"[IDS] Colunas removidas: {self.removed_columns}")
        print(f"[IDS] Features usadas no modelo: {self.feature_columns}")
        print(
            f"[IDS] Treino: {self._fitted_rows_before_undersampling} -> "
            f"{self._fitted_rows_after_undersampling} linhas após undersampling "
            f"({self.undersampling})"
        )
        print(f"[IDS] Detector: {self.detector.key} (escala: {self.scaler})")

        # Épico E8 — o detector é resolvido no __init__ e treinado aqui, no
        # mesmo ponto do pipeline em que o Random Forest era construído antes:
        # depois de preprocessar (E6), selecionar features e reamostrar (E7).
        # ``self.model`` continua existindo apontando para o estimador do
        # sklearn: é o que ``get_shap_importances`` lê e o que chamadores
        # anteriores ao E8 inspecionam. Só a *leitura* é compatível — atribuir
        # ``evaluator.model = clf`` por fora virou no-op, porque as predições
        # passam por ``self.detector``. Nenhum chamador deste repositório faz
        # isso; se algum passar a fazer, o caminho é reconstruir o Detector.
        self.detector.fit(X_train, y_train)
        self.model = self.detector.estimator
        y_pred = self.detector.predict(X_test)

        metrics = self._compute_metrics(
            y_true=y_test,
            y_pred=y_pred,
            dataset_path=dataset_path,
            dataset_rows=len(df),
            dataset_columns=len(df.columns),
            evaluation_type="baseline_internal_test",
        )

        print("[IDS] Métricas do baseline:")
        print(metrics)

        return metrics

    def evaluate_variant(self, dataset_path: str) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Modelo ainda não treinado. Execute train_baseline() primeiro.")

        if self.label_column is None or self.preprocessor is None:
            raise RuntimeError("Coluna de classe ainda não definida.")

        print(f"[IDS] Testando variante com modelo baseline: {dataset_path}")

        df = pd.read_csv(dataset_path)

        if df.empty:
            raise ValueError("Dataset variante vazio.")

        if self.label_column not in df.columns:
            raise ValueError(f"Coluna de classe '{self.label_column}' não existe no dataset variante.")

        X = df.drop(columns=[self.label_column])
        y = df[self.label_column]

        X_prepared = self.preprocessor.transform(X)
        if self.feature_selector is not None:
            X_prepared = self.feature_selector.transform(X_prepared)
        y_encoded = self._transform_labels(y)

        y_pred = self.detector.predict(X_prepared)

        metrics = self._compute_metrics(
            y_true=y_encoded,
            y_pred=y_pred,
            dataset_path=dataset_path,
            dataset_rows=len(df),
            dataset_columns=len(df.columns),
            evaluation_type="variant_external_test",
        )

        print("[IDS] Métricas da variante:")
        print(metrics)

        return metrics

    def _find_label_column(self, df: pd.DataFrame) -> str:
        possible_names = [
            "class",
            "label",
            "classe",
            "target",
            "attack",
            "is_attack",
            "category",
            "type",
        ]

        lower_columns = {column.lower(): column for column in df.columns}

        for name in possible_names:
            if name.lower() in lower_columns:
                return lower_columns[name.lower()]

        return df.columns[-1]

    def _fit_transform_labels(self, y: pd.Series) -> pd.Series:
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y.astype(str))

        self.class_mapping = {
            int(index): str(label)
            for index, label in enumerate(self.label_encoder.classes_)
        }

        self.attack_label = self._detect_attack_label(self.class_mapping)

        print(f"[IDS] Mapeamento de classes: {self.class_mapping}")
        print(f"[IDS] Classe de ataque: {self.attack_label}")

        return pd.Series(y_encoded)

    def _transform_labels(self, y: pd.Series) -> pd.Series:
        if self.label_encoder is None:
            raise RuntimeError("LabelEncoder ainda não treinado.")

        known_labels = set(self.label_encoder.classes_)

        encoded_values = []
        for value in y.astype(str):
            if value in known_labels:
                encoded_values.append(int(self.label_encoder.transform([value])[0]))
            else:
                encoded_values.append(-1)

        return pd.Series(encoded_values)

    def _detect_attack_label(self, class_mapping: dict[int, str]) -> int:
        # 1) Âncora explícita: se sabemos qual é o rótulo de ataque (via registry),
        # casamos por ele — funciona para qualquer ataque (random_replay, grayhole,
        # poisoned_high_rate, ...), inclusive os que não contêm "attack" no nome.
        if self.target_attack_label is not None:
            target = self.target_attack_label.lower()
            for encoded_label, original_label in class_mapping.items():
                if original_label.lower() == target:
                    return encoded_label

        # 2) Heurística por substring (fallback histórico).
        for encoded_label, original_label in class_mapping.items():
            normalized = original_label.lower()

            if (
                "masquerade" in normalized
                or "attack" in normalized
                or "malicious" in normalized
                or normalized in ["1", "true", "ataque"]
            ):
                return encoded_label

        # 3) Último recurso: a classe não-"normal", se houver exatamente uma.
        non_normal = [
            enc for enc, orig in class_mapping.items() if orig.lower() != "normal"
        ]
        if len(non_normal) == 1:
            return non_normal[0]

        return max(class_mapping.keys())

    # ------------------------------------------------------------------ #
    # API estável de feature importances (issue #15)                     #
    #                                                                    #
    # Assinatura congelada — consumida pelo Analista (#12) e por         #
    # ``_compute_metrics`` (popula ``Metrics.top_feature_importances``). #
    # Formato de cada item: ``{"feature": str, "importance": float}``,   #
    # compatível com ``domain.metrics.FeatureImportance`` (extra=forbid).#
    # ------------------------------------------------------------------ #
    def get_feature_importances(self, top_n: int = 15) -> list[dict[str, Any]]:
        """Importâncias do detector treinado, em ordem decrescente.

        Disponível após ``train_baseline``; ``[]`` se o modelo ainda não foi
        treinado **ou** se o detector não expõe importância nenhuma
        (``svm_rbf``). A assinatura e o formato de item seguem congelados
        (issue #15); o que o E8 acrescenta é que o *significado* do número
        varia por detector — Gini para RF/DT, ``|coef_|`` para o SVM linear.
        ``DetectorManifest.importance_kind`` é quem diz qual dos dois é.
        """
        return self.detector.feature_importances(self.feature_columns, top_n=top_n)

    def get_shap_importances(
        self,
        dataset_path: str,
        top_n: int = 15,
        max_samples: int = 200,
    ) -> list[dict[str, Any]]:
        """Importâncias via SHAP (média de ``|SHAP|``) para a classe de ataque.

        Best-effort ("SHAP quando possível"): calculado sobre ``dataset_path``
        já transformado no mesmo espaço de features do baseline. O pacote
        ``shap`` é opcional (extra ``shap`` no ``pyproject``); quando ausente —
        ou em qualquer falha de cálculo — retorna ``[]`` sem quebrar o loop, e
        o Analista recai sobre ``get_feature_importances`` (Gini).

        Mesmo formato de item de ``get_feature_importances`` para consumo
        uniforme pelo M2 (#12).

        Épico E8: ``shap.TreeExplainer`` só explica modelo baseado em árvore,
        então detectores com ``supports_shap=False`` (os dois SVMs) retornam
        ``[]`` aqui pelo mesmo caminho best-effort de quando o pacote ``shap``
        está ausente — nunca um explicador errado aplicado a um modelo que ele
        não sabe ler.
        """
        if self.model is None or self.feature_columns is None or self.preprocessor is None:
            return []

        if not self.detector.supports_shap:
            print(
                f"[IDS] SHAP indisponível para o detector '{self.detector.key}' "
                "(TreeExplainer só cobre modelos de árvore)."
            )
            return []

        try:
            import numpy as np
            import shap  # opcional — instalar com `pip install adversarial-ids[shap]`
        except ImportError:
            print("[IDS] SHAP indisponível (pacote 'shap' não instalado) — usando só Gini.")
            return []

        try:
            df = pd.read_csv(dataset_path)

            if self.label_column and self.label_column in df.columns:
                df = df.drop(columns=[self.label_column])

            X = self.preprocessor.transform(df)
            if self.feature_selector is not None:
                X = self.feature_selector.transform(X)

            if len(X) > max_samples:
                X = X.sample(n=max_samples, random_state=self.random_state)

            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)

            attack_shap = self._shap_values_for_attack_class(shap_values, np)
            if attack_shap is None:
                return []

            mean_abs = np.abs(attack_shap).mean(axis=0)

            ranking = sorted(
                (
                    {"feature": feature, "importance": float(value)}
                    for feature, value in zip(self.feature_columns, mean_abs)
                ),
                key=lambda item: item["importance"],
                reverse=True,
            )

            return ranking[:top_n]
        except Exception as error:  # best-effort: nunca derruba o loop
            print(f"[IDS] SHAP falhou ({error}) — usando só Gini.")
            return []

    def _shap_values_for_attack_class(self, shap_values: Any, np: Any) -> Any | None:
        """Extrai a matriz ``(n_samples, n_features)`` da classe de ataque.

        Normaliza os vários formatos que ``TreeExplainer.shap_values`` devolve
        entre versões do ``shap``: lista por classe, tensor 3D
        ``(n_samples, n_features, n_classes)`` ou já 2D (binário reduzido).
        """
        classes = list(self.model.classes_) if self.model is not None else []
        if self.attack_label in classes:
            attack_index = classes.index(self.attack_label)
        else:
            attack_index = len(classes) - 1 if classes else 0

        if isinstance(shap_values, list):
            if not shap_values:
                return None
            index = attack_index if attack_index < len(shap_values) else -1
            return np.asarray(shap_values[index])

        array = np.asarray(shap_values)
        if array.ndim == 3:
            index = attack_index if attack_index < array.shape[2] else -1
            return array[:, :, index]

        return array

    def _compute_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        dataset_path: str,
        dataset_rows: int,
        dataset_columns: int,
        evaluation_type: str,
    ) -> dict[str, Any]:
        if self.attack_label is None:
            raise RuntimeError("Classe de ataque não definida.")

        accuracy = accuracy_score(y_true, y_pred)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[self.attack_label],
            average=None,
            zero_division=0,
        )

        labels = sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        tp = fp = fn = tn = None

        if self.attack_label in labels and len(labels) == 2:
            attack_index = labels.index(self.attack_label)
            normal_index = 1 - attack_index

            tp = int(cm[attack_index][attack_index])
            fn = int(cm[attack_index][normal_index])
            fp = int(cm[normal_index][attack_index])
            tn = int(cm[normal_index][normal_index])

        attack_count = int((y_true == self.attack_label).sum())
        normal_count = int(len(y_true) - attack_count)

        return {
            "evaluation_type": evaluation_type,
            "dataset_path": dataset_path,
            "accuracy": float(accuracy),
            "attack_label_encoded": int(self.attack_label),
            "attack_label_original": self.class_mapping.get(int(self.attack_label)),
            "precision_attack": float(precision[0]),
            "recall_attack": float(recall[0]),
            "f1_score_attack": float(f1[0]),
            "support_attack": int(support[0]),
            "attack_count": attack_count,
            "normal_count": normal_count,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "label_column": self.label_column,
            "class_mapping": self.class_mapping,
            "dataset_rows": int(dataset_rows),
            "dataset_columns": int(dataset_columns),
            "removed_columns": self.removed_columns,
            "used_features": self.feature_columns,
            "top_feature_importances": self.get_feature_importances(),
        }