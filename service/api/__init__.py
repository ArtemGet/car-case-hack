"""service.api — тонкий FastAPI-фасад (W1-7).

HTTP/Swagger-слой поверх уже существующих модулей: он НЕ содержит логики
эмбеддинга и НЕ дублирует её. Эмбеддинг берётся у ``service.infer.backends``
(через :class:`service.api.engine.InferEngine`), ранжирование/уверенность — у
``reid.rerank``/``reid.calibrate``, векторный поиск — у статической галереи
(:mod:`service.api.gallery`).

Публичный вход: :func:`service.api.app.create_app`.
"""
from __future__ import annotations

from .app import create_app

__all__ = ["create_app"]
