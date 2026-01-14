from pyswip import Prolog

class KnowledgeBase:
    def __init__(self, kb_path: str):
        self.prolog = Prolog()
        self.prolog.consult(kb_path)

    @staticmethod
    def _to_prolog_str(s: str) -> str:
        return str(s).replace("'", "\\'")

    def get_reasons(self, row: dict, proba: float):
        q = (
            "reasons("
            f"'{self._to_prolog_str(row['country'])}', "
            f"'{self._to_prolog_str(row['bin_country'])}', "
            f"{float(row['amount'])}, "
            f"{float(row['shipping_distance_km'])}, "
            f"{int(row['transaction_hour'])}, "
            f"{int(row['avs_match'])}, "
            f"{int(row['cvv_result'])}, "
            f"{int(row['three_ds_flag'])}, "
            f"{int(row['high_amount'])}, "
            f"{float(proba)}, Reasons)."
        )
        res = list(self.prolog.query(q))
        return res[0]["Reasons"] if res else []

    def is_risky(self, row: dict, proba: float):
        q = (
            "risk_assess("
            f"'{self._to_prolog_str(row['country'])}', "
            f"'{self._to_prolog_str(row['bin_country'])}', "
            f"{float(row['amount'])}, "
            f"{float(row['shipping_distance_km'])}, "
            f"{int(row['transaction_hour'])}, "
            f"{int(row['avs_match'])}, "
            f"{int(row['cvv_result'])}, "
            f"{int(row['three_ds_flag'])}, "
            f"{int(row['high_amount'])}, "
            f"{float(proba)}, Reasons, ScoreRaw, ScoreNorm, Level)."
        )

        res = list(self.prolog.query(q))
        if res:
            reasons = res[0]["Reasons"]
            score_raw = int(res[0]["ScoreRaw"])
            score_norm = int(res[0]["ScoreNorm"])
            level = str(res[0]["Level"]).upper()
            if level == "HIGH":
                action = "BLOCKED"
            elif level == "MEDIUM":
                action = "REVIEW"
            else:
                action = "LEGITIMATE"

            return level, score_raw, score_norm, reasons, action

        return "LOW", 0, 0, [], "LEGITIMATE"
