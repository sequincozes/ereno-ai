"""ERENO AI Studio — camada de UI (Streamlit) do loop adversarial.

Frontend completo e temático (sala de controle de subestação) que substitui o
uso via terminal: configuração de parâmetros, execução com logs ao vivo e o
dashboard final de resultados — tudo pela interface.

O pacote é puramente de apresentação: reusa os *runners* de
``interfaces.experiment_runner`` e os modelos de ``domain`` sem duplicar lógica.
"""

from __future__ import annotations

__all__ = ["PALETTE"]

# Paleta central compartilhada por CSS, componentes e gráficos.
# Séries de dados (blue/orange/aqua) validadas para superfície escura pela
# skill de dataviz (validate_palette.js, --mode dark).
PALETTE = {
    "bg": "#0a0e14",
    "surface": "#0f1620",
    "surface_2": "#131c28",
    "surface_3": "#1a2432",
    "border": "rgba(120,160,200,0.14)",
    "border_strong": "rgba(120,160,200,0.30)",
    "text": "#e6edf3",
    "text_muted": "#8b98a9",
    "text_dim": "#5c6b7d",
    # Identidade de time (chrome da UI, não codificação de série)
    "red": "#ff4655",      # Red Team
    "red_soft": "rgba(255,70,85,0.14)",
    "cyan": "#34d3ee",     # Blue Team
    "cyan_soft": "rgba(52,211,238,0.14)",
    "amber": "#f5a524",
    # Status (reservados; sempre com ícone/rótulo)
    "good": "#35c07a",
    "warn": "#f5a524",
    "bad": "#ff4655",
    # Séries de dados validadas (dataviz)
    "series_1": "#3987e5",  # blue
    "series_2": "#d95926",  # orange
    "series_3": "#199e70",  # aqua
}
