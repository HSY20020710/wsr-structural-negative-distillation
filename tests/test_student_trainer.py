from __future__ import annotations

import unittest


class StudentTrainerTests(unittest.TestCase):
    def test_vectorized_per_sample_nll_matches_manual_formula(self) -> None:
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError:
            self.skipTest("torch is not installed")

        torch.manual_seed(7)
        logits = torch.randn(2, 5, 11)
        labels = torch.tensor(
            [
                [-100, -100, 3, 4, 5],
                [-100, 2, 1, -100, 6],
            ]
        )
        shifted_logits = logits[:, :-1].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        token_losses = functional.cross_entropy(
            shifted_logits.transpose(1, 2),
            shifted_labels,
            ignore_index=-100,
            reduction="none",
        )
        mask = shifted_labels.ne(-100)
        actual = (token_losses * mask).sum(dim=1) / mask.sum(
            dim=1
        ).clamp_min(1)

        expected = []
        for sample_logits, sample_labels in zip(
            shifted_logits, shifted_labels
        ):
            sample_mask = sample_labels.ne(-100)
            expected.append(
                functional.cross_entropy(
                    sample_logits[sample_mask],
                    sample_labels[sample_mask],
                    reduction="mean",
                )
            )
        self.assertTrue(torch.allclose(actual, torch.stack(expected)))


if __name__ == "__main__":
    unittest.main()
