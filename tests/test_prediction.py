"""Verify the saved dual-output contract and fold-local boundary behavior."""
import io
import json
import subprocess
import sys
import unittest

import joblib
import numpy as np
import pandas as pd

from scripts.audit_intake import ROOT
from scripts.predict import BUNDLE, predict_batch, predict_customer, predict_outputs
from scripts.spend_bands import LABELS, assign_bands, infer_cutoffs

EXAMPLE=dict(Age=35,Annual_Income=700000,Membership_Type="Silver",City_Tier="Tier 2")


class PredictionTests(unittest.TestCase):
    def test_boundaries(self):
        for i,c in enumerate(joblib.load(BUNDLE)["cutoffs"]):
            self.assertEqual(assign_bands([np.nextafter(c,-np.inf),c,np.nextafter(c,np.inf)],
                                         joblib.load(BUNDLE)["cutoffs"]).tolist(),
                             [LABELS[i],LABELS[i+1],LABELS[i+1]])

    def test_inferred_boundaries_and_conflicts(self):
        y=np.array([1,3,5,7,9])
        self.assertEqual(infer_cutoffs(y,LABELS),[2,4,6,8])
        with self.assertRaises(ValueError):
            infer_cutoffs([1,3,2,7,9],LABELS)

    def test_single_batch_reload(self):
        expected=predict_customer(EXAMPLE)
        actual=json.loads(subprocess.check_output([sys.executable,"-m","scripts.predict","--dual",
                          "--row",json.dumps(EXAMPLE)],cwd=ROOT,text=True))
        self.assertEqual(expected,actual)
        result=predict_batch(io.StringIO(pd.DataFrame([EXAMPLE,EXAMPLE]).assign(ID=["002","001"]).to_csv(index=False)),dual=True)
        self.assertEqual(result.ID.tolist(),["002","001"])
        np.testing.assert_allclose(result.Predicted_Amount_Spent,expected["Predicted_Amount_Spent"])

    def test_missing_and_unseen(self):
        value=predict_outputs(pd.DataFrame([dict(EXAMPLE,Age=None,Membership_Type="Unseen")]))
        self.assertTrue(np.isfinite(value.Predicted_Amount_Spent).all())
        with self.assertRaises(ValueError):
            predict_outputs(pd.DataFrame([{"Age":30}]))
        with self.assertRaises(ValueError):
            predict_outputs(pd.DataFrame([dict(EXAMPLE,Annual_Income="invalid")]))

    def test_inputs_exclude_targets(self):
        schema=joblib.load(BUNDLE)["schema"]
        self.assertFalse({"Amount_Spent","Amount_Spent_Range","Age_Group"} & set(schema["required"]))


if __name__=="__main__":
    unittest.main()
