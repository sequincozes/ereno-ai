"""Stubs determinísticos para rodar o loop sem os agentes reais (issue #3).

- FakeStrategist — stand-in do Red Team (permite M2 rodar sem o Estrategista).
- FakeAnalyst    — stand-in do Blue Team (permite M1 rodar sem o Analista).
"""

from tests.fakes.fake_analyst import FakeAnalyst
from tests.fakes.fake_strategist import FakeStrategist

__all__ = ["FakeStrategist", "FakeAnalyst"]
