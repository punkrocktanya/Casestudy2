"""Regression checks for the corrective-action interval and quality rules."""
import unittest
from datetime import time, timedelta
import numpy as np
import pandas as pd

import battery_validated_pipeline as pipeline


def samples(times=(0,1,2,3), currents=(1,1,-1,-1), phases=('CHARGE','CHARGE','DISCHARGE','DISCHARGE'),
            ids=(1,1,2,2), calendar=None):
    count=len(times)
    return pd.DataFrame({
        'Row_ID':range(1,count+1),'Data_Point':range(1,count+1),'Test_Time(s)':times,
        'Current(A)':currents,'Voltage(V)':np.repeat(3.7,count),
        'Capacity(Ah)':[0,1/3600,0,1/3600][:count],
        'Energy(Wh)':[0,3.7/3600,0,3.7/3600][:count],
        'Temperature(℃)':np.repeat(np.nan,count),
        'Date_Time':pd.to_datetime(calendar) if calendar else pd.date_range('2020-01-01',periods=count,freq='s'),
        'Cycle_Index':np.repeat(2,count),'Phase':phases,'Phase_ID':ids})


class IntervalTests(unittest.TestCase):
    def test_known_trapezoids_and_units(self):
        d=pipeline.build_intervals(samples())
        rec,_=pipeline.summarize_cycle('test',d,1,3,True)
        self.assertAlmostEqual(rec['Charge_Capacity_Ah'],1/3600)
        self.assertAlmostEqual(rec['Charge_Energy_Wh'],3.7/3600)
        self.assertAlmostEqual(rec['Mean_C_Rate_Charge'],0.4)
        self.assertAlmostEqual(rec['Coulombic_Efficiency_pct'],100)
        self.assertAlmostEqual(rec['Energy_Throughput_Wh'],7.4/3600)
        self.assertEqual(rec['Phase_Boundary_Positive_s'],1)
        self.assertEqual(rec['Quality_Status'],'REVIEW')
        self.assertTrue(np.isnan(rec['Protocol_Verified_CE_pct']))
        self.assertNotIn('Rest_Counter_Ah',rec)

    def test_calendar_reversal_keeps_valid_test_interval(self):
        raw=samples(calendar=['2020-01-01 00:00:01','2020-01-01 00:00:00','2020-01-01 00:00:02','2020-01-01 00:00:03'])
        d=pipeline.build_intervals(raw)
        self.assertTrue(d.Calendar_Reversal.iloc[1])
        self.assertTrue(d.Valid_Capacity.iloc[1])

    def test_zero_dt_is_not_lost_positive_time(self):
        d=pipeline.build_intervals(samples(times=(0,0,1,2)))
        self.assertTrue(d.Zero_dt.iloc[1]); self.assertFalse(d.Valid_Capacity.iloc[1])
        rec,_=pipeline.summarize_cycle('test',d,1,3,True)
        self.assertEqual(rec['Charge_Omitted_Positive_s'],0)

    def test_within_phase_gap_is_measured(self):
        d=pipeline.build_intervals(samples(times=(0,10,11,12)))
        rec,_=pipeline.summarize_cycle('test',d,1,3,True)
        self.assertEqual(rec['Charge_Omitted_Positive_s'],10)
        self.assertEqual(rec['Charge_Time_Coverage_pct'],0)
        self.assertFalse(rec['Charge_Integration_Valid'])
        self.assertEqual(rec['Quality_Status'],'FAIL')

    def test_phase_entry_gap_separate_from_phase_validity(self):
        d=pipeline.build_intervals(samples(times=(0,1,20,21)))
        rec,_=pipeline.summarize_cycle('test',d,1,3,True)
        self.assertTrue(rec['Discharge_Integration_Valid'])
        self.assertEqual(rec['Discharge_Time_Coverage_pct'],100)
        self.assertEqual(rec['Phase_Boundary_Positive_s'],19)
        self.assertIn('TEST_CLOCK_GAP',rec['Quality_Flags'])

    def test_missing_voltage_reduces_energy_coverage(self):
        raw=samples(); raw.loc[1,'Voltage(V)']=np.nan
        d=pipeline.build_intervals(raw)
        self.assertTrue(d.Valid_Capacity.iloc[1]); self.assertFalse(d.Valid_Energy.iloc[1])
        rec,_=pipeline.summarize_cycle('test',d,1,3,True)
        self.assertEqual(rec['Charge_Energy_Omitted_Positive_s'],1)
        self.assertFalse(rec['Charge_Integration_Valid'])

    def test_artifacts_exclude_touching_intervals_without_relabelling(self):
        raw=samples(); d=pipeline.build_intervals(raw,[False,True,False,False])
        self.assertFalse(d.Valid_Capacity.iloc[1])
        pd.testing.assert_series_equal(raw.Phase_ID,d.Phase_ID)
        pd.testing.assert_series_equal(raw.Phase,d.Phase)

    def test_counter_negative_jump_not_hidden_by_range(self):
        rows=pd.DataFrame({'Phase_ID':[1,1,1,1], 'Capacity(Ah)':[0,.2,.05,.15]})
        result=pipeline.counter_analysis(rows,'Capacity(Ah)')
        self.assertEqual(result['negative_jump_count'],1)
        self.assertAlmostEqual(result['positive_increment_total'],.3)

    def test_efficiency_above_100_flagged_not_clamped(self):
        raw=samples(currents=(1,1,-1.01,-1.01))
        d=pipeline.build_intervals(raw); rec,_=pipeline.summarize_cycle('test',d,1,3,True)
        self.assertAlmostEqual(rec['Coulombic_Efficiency_pct'],101)
        self.assertIn('CE_ABOVE_REVIEW_THRESHOLD',rec['Quality_Flags'])
        self.assertEqual(rec['Quality_Status'],'REVIEW')

    def test_incomplete_cycle_has_no_efficiency(self):
        raw=samples(currents=(1,1,1,1),phases=('CHARGE',)*4,ids=(1,1,1,1))
        d=pipeline.build_intervals(raw); rec,_=pipeline.summarize_cycle('test',d,2,2,True)
        self.assertTrue(rec['Partial_Charge'])
        self.assertTrue(np.isnan(rec['Coulombic_Efficiency_pct']))

    def test_negative_test_clock_prevents_valid_pair(self):
        d=pipeline.build_intervals(samples(times=(1,0,1,2)))
        self.assertTrue(d.Negative_dt.iloc[1]); self.assertFalse(d.Valid_Capacity.iloc[1])
        rec,_=pipeline.summarize_cycle('test',d,1,3,True)
        self.assertFalse(rec['Computationally_Complete'])

    def test_source_order_contract(self):
        raw=samples(); raw.loc[0,'Row_ID']=5
        with self.assertRaises(ValueError): pipeline.build_intervals(raw)

    def test_time_parser_over_24_hours(self):
        parser=pipeline.load_script('01_normalize_timestamps.py').time_to_seconds
        self.assertEqual(parser('1-01:02:03'),90123)
        self.assertEqual(parser('25:02:03'),90123)
        self.assertEqual(parser(timedelta(days=1,hours=1)),90000)
        self.assertEqual(parser(time(23,59,59)),86399)
        self.assertEqual(parser(90000),90000)

    def test_saved_dataset_contracts(self):
        path=pipeline.ROOT/'outputs'/'gold'/'battery_cycle_features_gold.parquet'
        if not path.exists(): self.skipTest('Build outputs first')
        try:
            gold=pd.read_parquet(path)
        except ImportError:
            # Parquet backend is an optional runtime dependency. The identical
            # Gold CSV is packaged and supports the same data-contract checks.
            csv=path.with_suffix('.csv')
            if not csv.exists(): self.skipTest('Install pyarrow to read saved Gold Parquet')
            gold=pd.read_csv(csv, dtype={'Dataset_ID': str})
        if 'Computationally_Complete' not in gold: self.skipTest('Legacy output; rebuild first')
        self.assertEqual(len(gold),202)
        self.assertFalse(gold.duplicated(['Dataset_ID','Cycle_Index']).any())
        for ds,sub in gold.groupby('Dataset_ID'):
            self.assertEqual(sub.Cycle_Index.tolist(),list(range(1,102)))
            self.assertEqual(sub.loc[sub.Coulombic_Efficiency_pct.isna(),'Cycle_Index'].tolist(),[1,101])
        self.assertEqual(int(gold.Coulombic_Efficiency_pct.gt(100).sum()),6)
        self.assertTrue(gold.loc[gold.Coulombic_Efficiency_pct.gt(100),'Quality_Flags'].str.contains('CE_ABOVE_REVIEW_THRESHOLD').all())
        self.assertFalse(gold.Full_Cycle_Protocol_Verified.any())
        self.assertFalse(gold.SOH_Training_Eligible.any())
        self.assertTrue(gold.Protocol_Verified_CE_pct.isna().all())


if __name__=='__main__': unittest.main()
