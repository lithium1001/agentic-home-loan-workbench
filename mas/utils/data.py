"""DataStore — wraps the star-schema CSV tables in ../csv_tables/.

Each public method corresponds exactly to one tool exposed to agents:

  get_profile()          → dim_customers.csv
  get_loan_application() → fact_loan_applications.csv
  get_bank_credit()      → fact_bank_internal.csv
  get_cpf_history()      → fact_cpf_history.csv
  get_income_docs()      → fact_income_docs.csv
  get_property_docs()    → fact_property_docs.csv

All tables share applicant_id (APP0001–APP0020) as the join key.
DataFrames are loaded once and kept in _cache for the process lifetime.
"""

import os

import pandas as pd


class DataStore:

    def __init__(self, data_dir: str):
        self._dir = data_dir
        self._cache: dict[str, pd.DataFrame] = {}

    def _load(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            path = os.path.join(self._dir, f"{name}.csv")
            self._cache[name] = pd.read_csv(path)
        return self._cache[name]

    def list_applicants(self) -> list[str]:
        return self._load("dim_customers")["applicant_id"].tolist()

    # ── tool-level accessors — called by execute_tool() in tools.py ───────────

    def get_profile(self, applicant_id: str) -> dict:
        aid = applicant_id.upper()
        dim  = self._load("dim_customers")
        row  = dim[dim["applicant_id"] == aid]
        if row.empty:
            return {"error": f"{aid} not found"}
        return row.iloc[0].to_dict()

    def get_loan_application(self, applicant_id: str) -> dict:
        aid  = applicant_id.upper()
        app  = self._load("fact_loan_applications")
        row  = app[app["applicant_id"] == aid]
        if row.empty:
            return {"error": f"No loan application for {aid}"}
        return row.iloc[0].to_dict()

    def get_bank_credit(self, applicant_id: str) -> dict:
        aid  = applicant_id.upper()
        bank = self._load("fact_bank_internal")
        row  = bank[bank["applicant_id"] == aid]
        if row.empty:
            return {"error": f"No bank/credit record for {aid}"}
        return row.iloc[0].to_dict()

    def get_cpf_history(self, applicant_id: str, months: int = 6) -> list[dict]:
        aid  = applicant_id.upper()
        cpf  = self._load("fact_cpf_history")
        rows = cpf[cpf["applicant_id"] == aid].sort_values(
            "contribution_month", ascending=False
        ).head(months)
        return rows.to_dict(orient="records")

    def get_income_docs(self, applicant_id: str) -> list[dict]:
        aid  = applicant_id.upper()
        inc  = self._load("fact_income_docs")
        rows = inc[inc["applicant_id"] == aid]
        return rows.to_dict(orient="records")

    def get_property_docs(self, applicant_id: str) -> list[dict]:
        aid  = applicant_id.upper()
        prop = self._load("fact_property_docs")
        rows = prop[prop["applicant_id"] == aid]
        return rows.to_dict(orient="records")

    # ── loan-package catalog (dim_loan_packages) — bank-wide, not per-applicant ──
    # Rates are uniform for all borrowers (prototype assumption). Used by the
    # compare_packages tool to price fixed vs floating (and own vs competitor)
    # by feeding each package's indicative_rate_pct into calculate_loan.

    @staticmethod
    def _clean(record: dict) -> dict:
        """Replace pandas NaN with None so the dict is valid JSON (NaN is not).
        Float NaN survives df.where(), so we coerce per-value here instead."""
        return {k: (None if pd.isna(v) else v) for k, v in record.items()}

    def list_loan_packages(self) -> list[dict]:
        """All available loan packages. NaN cells (e.g. spread on a fixed pkg)
        become None so the result serialises cleanly to JSON."""
        pkgs = self._load("dim_loan_packages")
        return [self._clean(r) for r in pkgs.to_dict(orient="records")]

    def get_loan_package(self, package_id: str) -> dict:
        pid  = package_id.upper()
        pkgs = self._load("dim_loan_packages")
        row  = pkgs[pkgs["package_id"].str.upper() == pid]
        if row.empty:
            return {"error": f"No loan package {pid}"}
        return self._clean(row.iloc[0].to_dict())
