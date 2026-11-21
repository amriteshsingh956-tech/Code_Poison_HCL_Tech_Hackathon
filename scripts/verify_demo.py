"""Exercise Streamlit widgets and compare their predictions with CLI outputs."""
import json
import subprocess
import sys
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from .audit_intake import ROOT


def main():
    """Submit the form and upload valid and invalid files through Streamlit AppTest."""
    ui = AppTest.from_file(str(ROOT/"app.py"), default_timeout=60).run()
    assert not ui.exception
    example = dict(Age=35, Annual_Income=700000, Membership_Type="Silver", City_Tier="Tier 2")
    cli = [sys.executable, "-m", "scripts.predict"]
    estimate = json.loads(subprocess.check_output(cli+["--dual","--row", json.dumps(example)], cwd=ROOT, text=True))
    ui.button[0].click().run()
    assert ui.metric[0].value == f"{estimate['Predicted_Amount_Spent']:,.2f}"
    assert ui.metric[1].value == estimate["Predicted_Amount_Spent_Range"]
    print("Single form matches CLI:", estimate, flush=True)
    with TemporaryDirectory() as tmp:
        src, dst = Path(tmp)/"input.csv", Path(tmp)/"out.csv"
        batch = pd.DataFrame([example,dict(example, Membership_Type="Other"),
                              dict(example, Age=None, City_Tier=None)])
        batch.insert(0,"customer_id",["003","001","002"])
        batch.to_csv(src,index=False)
        subprocess.run(cli+["--dual","--csv",str(src),"--out",str(dst)],cwd=ROOT,check=True)
        ui.file_uploader[0].upload("profiles.csv",src.read_bytes(),"text/csv").run()
        actual = ui.dataframe[0].value
        expected = pd.read_csv(dst,dtype={"customer_id":str})
        assert actual.customer_id.tolist()==expected.customer_id.tolist()==["003","001","002"]
        np.testing.assert_allclose(actual.Predicted_Amount_Spent,expected.Predicted_Amount_Spent,rtol=0,atol=1e-8)
        assert actual.Predicted_Amount_Spent_Range.tolist()==expected.Predicted_Amount_Spent_Range.tolist()
        assert ui.get("download_button") and not ui.exception
        for frame, message in [(batch.drop(columns="Age"),"Age"),(batch.assign(Age="abc"),"data row 1")]:
            ui.file_uploader[0].set_value(None).run()
            ui.file_uploader[0].upload("invalid.csv",frame.to_csv(index=False).encode(),"text/csv").run()
            assert message in ui.error[0].value and not ui.get("download_button") and not ui.exception
        print("Batch parity, ID order, download control, unseen category, missing values, "
              "missing header and invalid numeric input passed.")


if __name__ == "__main__":
    main()
