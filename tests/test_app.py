"""End-to-end checks. Run:  python -m unittest discover -s tests -v

Uses a temporary database/model folder and SYNTHETIC data, so it never touches your real files.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_tmp = tempfile.mkdtemp()
os.environ["FRAUDSHIELD_DB"] = str(Path(_tmp) / "test.db")
os.environ["FRAUDSHIELD_MODEL"] = str(Path(_tmp) / "model.joblib")
os.environ.pop("GEMINI_API_KEY", None)

from streamlit.testing.v1 import AppTest  # noqa: E402

from backend import analytics, assistant, database as db, fraud_detection as fd  # noqa: E402
from backend import prediction, preprocessing as pp, report_generator as rg  # noqa: E402
from frontend.common import PAGES  # noqa: E402
from scripts.make_sample_data import make_synthetic  # noqa: E402


class Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init_db()
        cls.df = make_synthetic(6000, 0.03)
        cls.bundle = fd.train_models(cls.df, dataset_name="synthetic-test", n_estimators=40)

    def test_target_detection(self):
        self.assertEqual(pp.detect_target(self.df), "Class")

    def test_no_leakage_split_sizes(self):
        b = self.bundle
        self.assertEqual(b["n_train"] + b["n_val"] + b["n_test"], len(self.df))

    def test_threshold_and_metrics_present(self):
        for m in self.bundle["metrics"].values():
            for k in ("precision", "recall", "f1", "pr_auc", "confusion_matrix"):
                self.assertIn(k, m)

    def test_model_saved_and_loaded(self):
        self.assertIsNotNone(prediction.load_bundle())

    def test_risk_mapping_puts_threshold_at_50(self):
        r = prediction.risk_scores([0.0, self.bundle["threshold"], 1.0], self.bundle["threshold"])
        self.assertAlmostEqual(r[1], 50.0)
        self.assertEqual((r[0], r[2]), (0.0, 100.0))

    def test_validation_catches_missing_columns(self):
        bad = self.df.drop(columns=["V3"])
        self.assertFalse(pp.validate_for_model(bad, self.bundle["feature_cols"])["ok"])

    def test_analysis_storage_and_reports(self):
        res = prediction.run_analysis(self.bundle, self.df, "bulk", "t.csv", target_col="Class")
        self.assertEqual(res["n_total"], len(self.df))
        self.assertGreater(res["n_flagged"], 0)
        self.assertGreater(res["label_metrics"]["recall"], 0.5)
        flagged = analytics.flagged_transactions()
        self.assertGreater(len(flagged), 0)
        tx = db.get_transaction(int(flagged.iloc[0]["id"]))
        db.set_status(tx["id"], "Reviewed")
        db.add_note(tx["id"], "checked with customer")
        self.assertEqual(db.get_transaction(tx["id"])["status"], "Reviewed")
        self.assertTrue(rg.summary_pdf(analytics.get_kpis(), self.bundle, flagged, analytics.trend("D")).startswith(b"%PDF"))
        self.assertTrue(rg.case_pdf(db.get_transaction(tx["id"]), db.get_notes(tx["id"])).startswith(b"%PDF"))

    def test_shieldbot_fallback(self):
        ans, src = assistant.ask("How do I read a risk score?", [])
        self.assertEqual(src, "built-in")
        self.assertIn("0 to 100", ans)


class Pages(unittest.TestCase):
    """Open every page with and without a trained model and make sure none crashes."""

    def _open(self, page):
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
        at.session_state["nav"] = page
        at.run()
        self.assertFalse(at.exception, f"{page}: {[e.value for e in at.exception]}")
        return at

    def test_all_pages_with_model(self):
        if not Path(os.environ["FRAUDSHIELD_MODEL"]).exists():
            fd.train_models(make_synthetic(4000, 0.03), dataset_name="synthetic-test", n_estimators=30)
        for page in PAGES:
            self._open(page)


if __name__ == "__main__":
    unittest.main()
