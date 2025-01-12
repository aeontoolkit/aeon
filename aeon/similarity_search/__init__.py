"""Similarity search module."""

__all__ = ["BaseSimilaritySearch", "QuerySearch", "SeriesSearch"]

from aeon.similarity_search._query_search import QuerySearch
from aeon.similarity_search._series_search import SeriesSearch
from aeon.similarity_search.base import BaseSimilaritySearch
