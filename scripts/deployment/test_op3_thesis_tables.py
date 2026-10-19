"""Verify reported FPS, sample SD and missing-condition handling from raw records."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from build_op3_thesis_tables import build, stats


class MeasurementTests(unittest.TestCase):
    def test_sample_sd(self):
        self.assertAlmostEqual(stats([1., 3.])['sd'], 2**.5)
        self.assertIsNone(stats([1.])['sd'])

    def test_error_exclusion_and_missing_walking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'run_metadata').mkdir()
            frames = root/'frames.csv'
            frames.write_text('error,latency_ms\n,10\nfailed,10000\n,20\n')
            resources = root/'resources.csv'
            resources.write_text('condition,cpu_usage_percent,memory_used_mb,memory_usage_percent,detector_rss_mb\ndiam,10,1000,30,200\ndiam,20,1200,40,220\n')
            report = {'condition':'diam', 'run_id':'run_01', 'completed':True,
                      'raw_measurements':{'timing_csv':str(frames), 'resource_csv':str(resources)},
                      'setup':{'benchmark_seconds_measured':2}, 'results':{'frames_processed':2}}
            (root/'run_metadata/run.json').write_text(json.dumps(report))
            runs, samples = build(root)
            self.assertFalse((root/'tables/table_3_2_resources_raw.md').exists())
            self.assertFalse((root/'resources/resources_all.md').exists())
            self.assertEqual((root/'tables/table_3_2_resources_raw.csv').read_bytes(),
                             (root/'resources/resources_all.csv').read_bytes())
            self.assertEqual(runs[0]['fps'], 1)
            self.assertEqual(runs[0]['latency_mean_ms'], 15)
            self.assertIsNone(samples[1]['cpu_usage_percent_mean'])
            self.assertEqual(samples[1]['samples'], 0)
            self.assertEqual(samples[0]['memory_used_mb_mean'], 1100)


if __name__ == '__main__':
    unittest.main()
