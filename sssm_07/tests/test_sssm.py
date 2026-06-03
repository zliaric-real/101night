# test_sssm.py
import unittest
import numpy as np
from ..sssm.sssm import Model
import mne
import pandas as pd

class TestSSSMUtilities(unittest.TestCase):
    def test_model(self):
        raw = mne.io.read_raw_bdf('D:/baoxiaoyu/python_part/sssm_07/tests/陈红英20251210_f3.bdf')
        raw = raw.pick_channels(['CH3'])
        raw.load_data()
        raw.filter(0.1, 40)
        raw = raw.resample(100)
        data = raw.get_data(units="uV")

        # model = Model()
        model = Model(device='cpu')

        model.predict(data,step=300)
        pred_labels = model.pred
        window_indices = np.arange(pred_labels.shape[1])  # 窗口索引
        df_raw = pd.DataFrame({
            'Window_Index': window_indices,
            'Label_Index': pred_labels[0]
        })
        df = model.to_pandas()
        df_raw.to_csv('predictions.csv', index=False)
        print(df)


if __name__ == "__main__":
    unittest.main()


