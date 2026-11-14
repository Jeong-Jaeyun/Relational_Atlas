"""Exact-binomial raw-count audit of the independent weak-null Aer rerun."""

from __future__ import annotations

import json
from relational_atlas_i01.paths import PROJECT_ROOT

from relational_atlas_i01.holdout import qiskit_rank_holdout3_audit as prior
from relational_atlas_i01.holdout import qiskit_rank_holdout4b_weaknull as corrected


RESULT = PROJECT_ROOT / "results" / "qiskit_rank_holdout4b_audit.json"


def main() -> None:
    source = json.loads(corrected.RESULT.read_text(encoding="utf-8"))
    output = prior.audit(source, panels=5000, seed=20261021)
    output["source"] = str(corrected.RESULT)
    output["caveat"] = (
        "Exact-binomial raw-count QC and uniform-target reweighting added after the independent weak-null tests; diagnostic only.")
    RESULT.write_text(json.dumps(output, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps({"standardized_count_mean": output["standardized_count_mean"],
                      "standardized_count_variance": output["standardized_count_variance"],
                      "null_mean_95pct_interval": output["null_mean_95pct_interval"],
                      "null_variance_95pct_interval":
                          output["null_variance_95pct_interval"],
                      "result": str(RESULT)}, indent=2))


if __name__ == "__main__":
    main()
