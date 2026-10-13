"""Pool values/gradients, native model equivalence and provenance guards."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from src.artifacts import read_json, write_json
from research_v4.models import build_model as original_build_model
from research_v4_mps_compat.pooling import ExactAdaptiveAvgPool1d
from research_v4_mps_compat import runtime
from tests.test_research_v4 import ROOT, LEGACY, NEW


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_values_and_gradients_match_native_adaptive_pooling(self):
        for dtype, tolerance in ((torch.float32, 1e-6), (torch.float64, 1e-12)):
            for length, bins in ((157, 16), (160, 16), (7, 16), (1, 16), (193, 1)):
                with self.subTest(dtype=dtype, length=length, bins=bins):
                    x = torch.randn(2, 3, length, dtype=dtype, requires_grad=True)
                    y = x.detach().clone().requires_grad_()
                    reference = nn.AdaptiveAvgPool1d(bins)(x)
                    actual = ExactAdaptiveAvgPool1d(bins)(y)
                    torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)
                    upstream = torch.randn_like(actual)
                    reference.backward(upstream); actual.backward(upstream)
                    torch.testing.assert_close(y.grad, x.grad, atol=tolerance, rtol=tolerance)

    def test_tail_and_unbatched_inputs_are_preserved(self):
        x = torch.zeros(2, 157); x[:, -1] = 11
        actual = ExactAdaptiveAvgPool1d(16)(x)
        torch.testing.assert_close(actual, nn.AdaptiveAvgPool1d(16)(x))
        self.assertTrue((actual[:, -1] > 0).all())
        self.assertEqual(ExactAdaptiveAvgPool1d(16).state_dict(), {})

    def test_all_other_models_and_parameters_unchanged(self):
        for name in ('encoder', 'patchtst', 'hierarchical', 'residual_multiscale', 'lead_time'):
            torch.manual_seed(8); base = original_build_model(name, 'attention', LEGACY, NEW)
            torch.manual_seed(8); compat = runtime.build_model(name, 'attention', LEGACY, NEW)
            self.assertEqual(base.state_dict().keys(), compat.state_dict().keys())
            for key in base.state_dict():
                self.assertTrue(torch.equal(base.state_dict()[key], compat.state_dict()[key]))
            if name == 'lead_time':
                self.assertIsInstance(compat.backbone.cnn[-1], ExactAdaptiveAvgPool1d)
            else:
                self.assertEqual(str(base), str(compat))

    def test_native_lead_time_patient_output_and_parameter_gradients_match(self):
        config = {**NEW, 'dimension': 64, 'temporal_bins': 16}
        for pooling in ('mean', 'attention'):
            torch.manual_seed(7)
            base = original_build_model('lead_time', pooling, LEGACY, config).eval()
            compat = runtime.build_model('lead_time', pooling, LEGACY, config).eval()
            compat.load_state_dict(base.state_dict(), strict=True)
            x = torch.randn(2, 2, 12, 5000)
            mask = torch.tensor([[True, False], [True, True]])
            reference, actual = base(x, mask), compat(x, mask)
            torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-6)
            labels = torch.tensor([0, 2])
            nn.NLLLoss()(reference, labels).backward(); nn.NLLLoss()(actual, labels).backward()
            for left, right in zip(base.parameters(), compat.parameters()):
                self.assertTrue(torch.isfinite(right.grad).all())
                torch.testing.assert_close(right.grad, left.grad, atol=3e-6, rtol=2e-5)

    def test_immutable_patch_and_case_audit_preserves_original_configs(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as tmp:
            run = Path(tmp)
            write_json(run / 'protocol.json', {'experiment_hash': 'fixture'})
            old = run / 'cv/encoder/mean/S04/fold_1/seed_42'
            write_json(old / 'metrics.json', {'done': True})
            case = run / 'cv/lead_time/mean/S04/fold_1/seed_42'
            write_json(case / 'config.json', {'device': 'mps', 'case_hash': 'frozen'})
            original = (case / 'config.json').read_bytes()
            runtime.bind_case(run, case)
            runtime.bind_case(run, case)
            self.assertEqual((case / 'config.json').read_bytes(), original)
            record = runtime.audit_run(run)
            self.assertIn(str((old / 'metrics.json').relative_to(run)), record['preserved_artifact_sha256_at_activation'])
            changed = {**runtime.patch_identity(run), 'version': 'different'}
            with patch.object(runtime, 'patch_identity', return_value=changed):
                with self.assertRaisesRegex(RuntimeError, 'compatibility changed'):
                    runtime.audit_run(run)
            sidecar = read_json(case / runtime.RECORD)
            sidecar['compatibility_hash'] = 'tampered'
            write_json(case / runtime.RECORD, sidecar)
            with self.assertRaisesRegex(RuntimeError, 'compatibility changed'):
                runtime.bind_case(run, case)

    def test_unbound_lead_time_progress_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as tmp:
            run = Path(tmp)
            write_json(run / 'protocol.json', {'experiment_hash': 'fixture'})
            write_json(run / 'cv/lead_time/mean/S04/fold_1/seed_42/metrics.json', {})
            with self.assertRaisesRegex(RuntimeError, 'Unbound LeadTime'):
                runtime.audit_run(run, create=True)

    def test_incomplete_run_rejects_test_before_original_evaluation(self):
        with patch.object(runtime, 'require_final_complete', side_effect=RuntimeError('pending')):
            with patch.object(runtime, 'ORIGINAL_EVALUATE') as evaluate:
                with self.assertRaisesRegex(RuntimeError, 'pending'):
                    runtime.evaluate(Path('unused'), None, {}, {}, torch.device('cpu'))
                evaluate.assert_not_called()

    def test_fit_hook_records_compatibility_before_original_training(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as tmp:
            run = Path(tmp)
            write_json(run / 'protocol.json', {'experiment_hash': 'fixture'})
            signature = runtime.inspect.signature(runtime.ORIGINAL_FIT)
            def fake(*args, **kwargs):
                bound = signature.bind(*args, **kwargs); bound.apply_defaults()
                case = runtime.case_directory(bound.arguments)
                self.assertTrue((case / runtime.RECORD).is_file())
                return {'done': True}
            fake.__signature__ = signature
            with patch.object(runtime, 'ORIGINAL_FIT', fake):
                result = runtime.fit_case(run, run, {}, {}, 'lead_time', 'mean', {'id': 'S04'},
                                          torch.device('mps'), fold=1)
            self.assertEqual(result, {'done': True})


if __name__ == '__main__':
    unittest.main()
