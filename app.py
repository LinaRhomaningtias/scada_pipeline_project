"""Local entry point for the SCADA Pipeline web app.

Run from the project root:
    uvicorn app:app --reload
"""
from api.index import app

__all__ = ["app"]
