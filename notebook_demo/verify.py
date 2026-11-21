"""Check inference validation and Streamlit form parity."""
import json
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from notebook_demo.model import OUT, ROOT, predict


def main():
    """Exercise representative local inputs and the actual Streamlit form."""
    bundle = joblib.load(OUT / "model.pkl")
    row = {"Age": 35, "Annual_Income": 700000, "Website_Visits": 10,
           "Previous_Purchases": 5, "Membership_Type": "Silver",
           "Preferred_Category": "Electronics", "City_Tier": "Tier 2", "Discount_Used": "Yes"}
    frame = pd.DataFrame([row])
    expected = predict(frame, bundle)["Predicted_Amount_Spent"].iloc[0]
    batch = pd.DataFrame([row, row]).assign(Customer_ID=["003", "001"])
    result = predict(batch, bundle)
    assert result.Customer_ID.tolist() == ["003", "001"]
    assert np.allclose(result.Predicted_Amount_Spent, expected)
    edge = frame.copy()
    edge.loc[0, "Age"] = np.nan
    edge.loc[0, "Membership_Type"] = "Unseen"
    assert np.isfinite(predict(edge, bundle).Predicted_Amount_Spent).all()
    for bad in [frame.drop(columns="Age"), frame.assign(Age="abc"), frame.assign(Age=-1), frame.assign(Age=np.inf)]:
        try:
            predict(bad, bundle)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid input accepted")
    code = "import json,joblib,pandas as pd; from notebook_demo.model import OUT,predict; print(predict(pd.DataFrame([json.loads(" + repr(json.dumps(row)) + ")]),joblib.load(OUT/'model.pkl')).Predicted_Amount_Spent.iloc[0])"
    fresh = float(subprocess.check_output([sys.executable, "-c", code], cwd=ROOT, text=True))
    assert np.isclose(fresh, expected, rtol=0, atol=1e-8)
    app = AppTest.from_file(str(ROOT / "notebook_demo" / "app.py"), default_timeout=30).run()
    assert not app.exception, app.exception
    for widget in app.number_input:
        widget.set_value(float(row[widget.key]))
    for widget in app.selectbox:
        widget.set_value(row[widget.key])
    app.button[0].click().run()
    assert not app.exception, app.exception
    shown = [m.value for m in app.metric if m.label == "Estimated customer spend"]
    assert shown == [f"{expected:,.2f}"], shown
    print(json.dumps({"checks": ["single", "batch/order/identifiers", "missing/unseen", "invalid schema/types/ranges", "fresh-process reload", "Streamlit form parity"], "example": row, "prediction": expected, "fresh_process_difference": fresh - expected}, indent=2))


if __name__ == "__main__":
    main()
