"""Numeric and rendering contracts for focused research figures."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scripts.reporting.figure_style import comparison, delta_plot, local_limits, plt
from scripts.deployment.plot_op3_resources import plot_resources
from scripts.reporting.generate_thesis_figures import Figures


class FigureContracts(unittest.TestCase):
    def test_flat_and_bounded_data_remain_visible(self):
        for data, bounds in [([66.34, 66.472], (0, 100)), ([100, 100], (0, 100)),
                             ([0, 0], (0, 100)), ([257.4, np.nan, 257.4], None)]:
            lo, hi = local_limits(data, bounds)
            finite = np.asarray(data)[np.isfinite(data)]
            self.assertLess(lo, hi)
            self.assertLessEqual(lo, finite.min())
            self.assertGreaterEqual(hi, finite.max())

    def test_absolute_bars_keep_zero_and_detail_preserves_observations(self):
        fig, axes = plt.subplots(1, 2)
        series = {'FP32': [93.13469, 95.05717], 'QAT': [93.41156, 95.02079]}
        comparison(*axes, ['Precision', 'Recall'], series, '%', 4, (0, 100))
        self.assertEqual(axes[0].get_ylim()[0], 0)
        np.testing.assert_allclose([p.get_height() for p in axes[0].patches],
                                   [93.13469, 95.05717, 93.41156, 95.02079])
        for collection, expected in zip(axes[1].collections, series.values()):
            np.testing.assert_allclose(collection.get_offsets()[:, 0], expected)
        plt.close(fig)

    def test_both_directions_and_percentage_point_units(self):
        fig, ax = plt.subplots()
        result = delta_plot(ax, ['mAP50', 'mAP50–95'], [96.423175, 77.990816],
                            [97.214526, 76.717862], 'poin persentase')
        np.testing.assert_allclose(result, [.791351, -1.272954], atol=1e-9)
        self.assertLess(ax.get_xlim()[0], 0)
        self.assertGreater(ax.get_xlim()[1], 0)
        plt.close(fig)

    def test_reject_missing_or_negative_magnitude_values(self):
        fig, axes = plt.subplots(1, 2)
        for values in [[1, np.nan], [1, -1]]:
            with self.assertRaises(ValueError):
                comparison(*axes, ['A', 'B'], {'Model': values}, 'unit')
        plt.close(fig)

    def test_resource_plot_preserves_input_and_missing_temperature(self):
        with tempfile.TemporaryDirectory() as temp:
            path, output = Path(temp) / 'samples.csv', Path(temp) / 'plot.png'
            data = ('elapsed_seconds,cpu_usage_percent,detector_process_cpu_percent,memory_used_mb,'
                    'detector_rss_mb,temperature_c,condition,run_id\n'
                    '0,10,120,4000,257.1,,test,run_01\n'
                    '1,11,121,4001,257.2,,test,run_01\n')
            path.write_text(data)
            plot_resources(path, output)
            self.assertEqual(path.read_text(), data)
            self.assertGreater(output.stat().st_size, 1000)
            with self.assertRaises(SystemExit):
                plot_resources(path, output)

    def test_completed_exports_decode_in_all_supported_formats(self):
        from PIL import Image
        from xml.etree import ElementTree
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            figures = Figures(root, root, ['png', 'svg', 'pdf'])
            fig, ax = plt.subplots()
            ax.plot([1, 2], [66.340, 66.472])
            figures.save(fig, 'export', ['measured.csv'], 'Scale test')
            with Image.open(root / 'export.png') as image:
                image.load()
            ElementTree.parse(root / 'export.svg')
            self.assertTrue((root / 'export.pdf').read_bytes().startswith(b'%PDF'))
            self.assertEqual(list(root.glob('*.tmp')), [])


if __name__ == '__main__':
    unittest.main()
