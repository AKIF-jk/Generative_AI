"""Tests for the Task 1 universal restoration autoencoder."""

from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class UniversalRestorationAutoencoderTests(unittest.TestCase):
    @staticmethod
    def _model_type():
        from pet_restoration import UniversalRestorationAutoencoder

        return UniversalRestorationAutoencoder

    def test_all_ablation_variants_return_finite_unit_range_rgb(self) -> None:
        import torch

        input_batch = torch.rand(2, 3, 128, 128)
        for skip_count in (0, 1, 2):
            with self.subTest(skip_count=skip_count):
                model = self._model_type()(base_channels=32, latent_dim=128, dropout=0.1, skip_count=skip_count).eval()
                with torch.no_grad():
                    output = model(input_batch)
                self.assertEqual(tuple(output.shape), (2, 3, 128, 128))
                self.assertTrue(torch.isfinite(output).all())
                self.assertGreaterEqual(float(output.min()), 0.0)
                self.assertLessEqual(float(output.max()), 1.0)

    def test_vector_width_and_disabled_skip_modules(self) -> None:
        model_type = self._model_type()
        model = model_type(base_channels=32, latent_dim=512, skip_count=0)
        self.assertEqual(model.to_latent.out_features, 512)
        self.assertEqual(model.from_latent.in_features, 512)
        self.assertEqual(model.pre_latent_shape, (512, 4, 4))
        self.assertIsNone(model.skip_16)
        self.assertIsNone(model.skip_32)
        self.assertFalse(any(key.startswith("skip_") for key in model.state_dict()))

    def test_skip_layout_has_only_compressed_16_and_32_routes(self) -> None:
        import torch

        model_type = self._model_type()
        one_skip = model_type(base_channels=32, latent_dim=128, skip_count=1)
        two_skips = model_type(base_channels=32, latent_dim=128, skip_count=2)
        self.assertEqual(one_skip.skip_16.encoder_channels, one_skip.channel_progression[2])
        self.assertEqual(one_skip.skip_16.decoder_channels, one_skip.channel_progression[2])
        self.assertEqual(one_skip.skip_16.compressed_channels, one_skip.channel_progression[2] // 4)
        self.assertIsNone(one_skip.skip_32)
        self.assertEqual(two_skips.skip_32.encoder_channels, two_skips.channel_progression[1])
        self.assertEqual(two_skips.skip_32.decoder_channels, two_skips.channel_progression[1])
        self.assertEqual(two_skips.skip_32.compressed_channels, two_skips.channel_progression[1] // 4)

        observed_shapes: dict[str, tuple[int, int]] = {}
        handles = [
            two_skips.skip_16.compress.register_forward_pre_hook(
                lambda _module, inputs: observed_shapes.__setitem__("16x16", tuple(inputs[0].shape[-2:]))
            ),
            two_skips.skip_32.compress.register_forward_pre_hook(
                lambda _module, inputs: observed_shapes.__setitem__("32x32", tuple(inputs[0].shape[-2:]))
            ),
        ]
        try:
            with torch.no_grad():
                two_skips(torch.rand(1, 3, 128, 128))
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(observed_shapes, {"16x16": (16, 16), "32x32": (32, 32)})

    def test_backward_reaches_all_enabled_gates(self) -> None:
        import torch

        for skip_count in (0, 1, 2):
            with self.subTest(skip_count=skip_count):
                model = self._model_type()(base_channels=32, latent_dim=128, skip_count=skip_count)
                model(torch.rand(1, 3, 128, 128)).mean().backward()
                self.assertIsNotNone(model.to_latent.weight.grad)
                if skip_count >= 1:
                    self.assertIsNotNone(model.skip_16.gate.grad)
                    self.assertTrue(torch.isfinite(model.skip_16.gate.grad).all())
                if skip_count >= 2:
                    self.assertIsNotNone(model.skip_32.gate.grad)
                    self.assertTrue(torch.isfinite(model.skip_32.gate.grad).all())

    def test_gate_inspection_is_detached_and_label_free(self) -> None:
        import torch

        model = self._model_type()(base_channels=32, latent_dim=128, skip_count=2)
        values = model.skip_gate_values()
        self.assertEqual(set(values), {"16x16", "32x32"})
        self.assertFalse(values["16x16"].requires_grad)
        self.assertEqual(values["16x16"].numel(), model.skip_16.compressed_channels)
        self.assertEqual(values["32x32"].numel(), model.skip_32.compressed_channels)
        expected = torch.full_like(values["16x16"], torch.sigmoid(torch.tensor(-2.0)))
        self.assertTrue(torch.allclose(values["16x16"], expected))
