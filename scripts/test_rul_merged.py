"""Tests for the replaced three early-RUL implementations and reporting lineage."""
import unittest
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import torch
from individual_curve_rul import INPUT,OUT as CURVE_OUT,MODELS, validate,fit_cell,curve
from estimate_rul_xgboost_shared import OUT as XGB_OUT, FEATURES, examples,features,validate_groups, forecast
from estimate_rul_lstm_minimal import OUT as LSTM_OUT, prepare, MinimalLSTM, train, recurse
ROOT=Path(__file__).resolve().parent.parent

class MergedRUL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=pd.read_csv(INPUT,dtype={'Dataset_ID':str})

    def test_two_cells_and_target_coverage(self):
        groups,eols=validate_groups(self.data)
        self.assertEqual(eols,{'001_2':13,'002_4':24})
        self.assertEqual({c:len(g) for c,g in groups.items()},{'001_2':99,'002_4':99})

    def test_curve_uses_only_checkpoint_history(self):
        g=self.data[self.data.Dataset_ID=='001_2'].copy()
        h=validate(g,8)
        changed=g.copy();changed.loc[changed.Cycle_Index>8,'Discharge_Capacity_Ah']*=4
        h2=validate(changed,8)
        for name in ['Linear','Exponential','Square-root cycle','Power law']:
            a=fit_cell(h,name,shared_b=1.0);b=fit_cell(h2,name,shared_b=1.0)
            np.testing.assert_allclose(a,b)
        self.assertEqual(len(h),7)

    def test_curve_eol_and_status_consistency(self):
        d=pd.read_csv(CURVE_OUT/'predictions.csv',dtype={'Dataset_ID':str})
        for r in d.itertuples():
            self.assertEqual(int(r.Actual_RUL),int(r.Actual_EOL-r.Checkpoint))
            if r.Status=='crossing_predicted':
                self.assertAlmostEqual(r.Predicted_RUL,r.Predicted_EOL-r.Checkpoint)
                self.assertAlmostEqual(r.Absolute_Error,abs(r.Predicted_RUL-r.Actual_RUL))
            else:self.assertTrue(pd.isna(r.Predicted_RUL))
        self.assertEqual(len(d),40)

    def test_xgboost_pools_within_cell_only(self):
        g=self.data[(self.data.Dataset_ID=='001_2') & (self.data.Cycle_Index<=8)]
        s=examples(g)
        self.assertEqual(len(s),4)
        self.assertEqual(int(s.Target_Cycle.max()),8)
        self.assertTrue((s.Target_Cycle>=5).all())
        self.assertEqual(list(s.columns),FEATURES+['Target_Change','Target_Cycle'])

    def test_xgboost_normalized_inputs_and_error_arithmetic(self):
        data=pd.read_csv(XGB_OUT/'predictions.csv',dtype={'Dataset_ID':str})
        self.assertEqual(len(data),8)
        self.assertTrue((data.Training_Windows==2*(data.Checkpoint_Cycle-4)).all())
        np.testing.assert_allclose(data.Actual_RUL_Cycles,data.Actual_EOL_Cycle-data.Checkpoint_Cycle)
        np.testing.assert_allclose(data.Predicted_RUL_Cycles,data.Predicted_EOL_Cycle-data.Checkpoint_Cycle)
        np.testing.assert_allclose(data.Absolute_Error_Cycles,abs(data.Signed_Error_Cycles))
        self.assertEqual(features([1,.98,.94])[:3],[.94,.98,1.0])

    def test_lstm_windows_are_checkpoint_limited(self):
        x,y,h,a=prepare(self.data,8,3)
        self.assertEqual(len(y),8)
        self.assertEqual(x.shape,(8,3,1))
        self.assertTrue(all(z['Target_Cycle']<=8 for z in a))
        self.assertEqual(set(h),{'001_2','002_4'})

    def test_lstm_parameter_count_and_determinism(self):
        m=MinimalLSTM()
        self.assertEqual(sum(p.numel() for p in m.parameters()),18)
        x,y,h,a=prepare(self.data,5,3)
        m1,_=train(x,y,42);m2,_=train(x,y,42)
        for aa,bb in zip(m1.parameters(),m2.parameters()):torch.testing.assert_close(aa,bb)

    def test_lstm_failed_forecasts_stay_unscored(self):
        d=pd.read_csv(LSTM_OUT/'predictions.csv',dtype={'Dataset_ID':str})
        self.assertEqual(len(d),48)
        self.assertEqual(int((d.Status=='no_crossing_within_horizon').sum()),2)
        self.assertTrue(d.loc[d.Status!='crossing_predicted','Predicted_RUL'].isna().all())
        scored=d[d.Status=='crossing_predicted']
        np.testing.assert_allclose(scored.Predicted_RUL,scored.Predicted_EOL-scored.Checkpoint)
        np.testing.assert_allclose(scored.Absolute_Error,abs(scored.Predicted_RUL-scored.Actual_RUL))

    def test_all_models_share_input_hash(self):
        h=hashlib.sha256(INPUT.read_bytes()).hexdigest()
        import json
        for path in [CURVE_OUT/'manifest.json',XGB_OUT/'manifest.json',LSTM_OUT/'manifest.json']:
            self.assertEqual(json.loads(path.read_text())['input_sha256'],h)

    def test_common_checkpoint_includes_all_seed_attempts(self):
        path=ROOT/'outputs/rul_common_checkpoint/predictions.csv'
        d=pd.read_csv(path,dtype={'Dataset_ID':str})
        self.assertEqual(set(d.Checkpoint),{8,10})
        self.assertEqual(len(d),24)
        self.assertEqual(len(d[d.Method=='Minimal LSTM (window 3)']),12)
        self.assertEqual(len(d[d.Method=='Shared XGBoost']),4)
        self.assertTrue(((d.Predicted_RUL-d.Actual_RUL).abs().fillna(0)-d.Absolute_Error.fillna(0)).abs().max()<1e-9)

    def test_all_seven_methods_reconciled_at_common_checkpoints(self):
        base=ROOT/'outputs/rul_common_checkpoint'
        attempts=pd.read_csv(base/'all_method_common_predictions.csv',dtype={'Dataset_ID':str})
        scores=pd.read_csv(base/'all_method_common_metrics.csv',dtype={'Dataset_ID':str})
        self.assertEqual(len(attempts),44)
        self.assertEqual(len(scores),14)
        self.assertEqual(set(attempts.Checkpoint),{8,10})
        self.assertTrue((scores.groupby('Dataset_ID').Method.nunique()==7).all())
        self.assertEqual(int(scores.Attempts.sum()),len(attempts))
        for r in scores.itertuples():
            t=attempts[(attempts.Dataset_ID==r.Dataset_ID)&(attempts.Method==r.Method)]
            self.assertEqual(len(t),int(r.Attempts))
            self.assertEqual(int(t.Scored.sum()),int(r.Scored))
            self.assertAlmostEqual(float(t.Absolute_Error.mean()),float(r.MAE_Cycles))

    def test_all_96_attempts_appear_in_reporting_source(self):
        normalized=pd.read_csv(ROOT/'outputs/rul_all_models_validation/all_96_attempts.csv',dtype={'Dataset_ID':str})
        self.assertEqual(len(normalized),96)
        self.assertEqual(len(normalized.Method.unique()),7)
        self.assertEqual(int(normalized.Predicted_RUL.isna().sum()),5)
        self.assertEqual(int(normalized.Scored.sum()),91)
        np.testing.assert_allclose(normalized.Predicted_RUL-normalized.Actual_RUL,normalized.Signed_Error,equal_nan=True)

    def test_no_older_rul_artifacts(self):
        for f in ['estimate_rul_lstm.py','estimate_rul_xgboost.py','estimate_rul_extrapolation.py']:
            self.assertFalse((ROOT/f).exists())
        for f in ['rul_lstm','rul_xgboost','rul_extrapolation']:
            self.assertFalse((ROOT/'outputs'/f).exists())

if __name__=='__main__': unittest.main()
