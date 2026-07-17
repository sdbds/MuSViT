import itertools
import unittest

import editdistance

from experiments.full_page_omr.eval.eval_functions import (
    CanonicalTokenStream,
    canonical_text,
    canonicalize_prediction_ids,
    canonicalize_target_ids,
    compute_canonical_metrics,
    levenshtein_legacy,
    metric_views,
)


I2W = {
    0: "<pad>",
    1: "<bos>",
    2: "<eos>",
    3: "note",
    4: "<s>",
    5: "ab",
    6: "<t>",
    7: "d",
    8: "<b>",
    9: "·",
    10: "c",
}


class CanonicalTokenStreamTests(unittest.TestCase):
    def test_target_removes_optional_bos_eos_and_padding(self):
        without_bos = canonicalize_target_ids([3, 4, 2, 0], I2W)
        with_bos = canonicalize_target_ids([1, 3, 4, 2, 0], I2W)

        expected = CanonicalTokenStream(("note", "<s>"), True, False)
        self.assertEqual(without_bos, expected)
        self.assertEqual(with_bos, expected)

    def test_target_without_eos_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ground truth.*<eos>"):
            canonicalize_target_ids([3, 4, 0], I2W)

    def test_target_rejects_padding_or_bos_inside_scored_content(self):
        for token_id in (0, 1):
            with self.subTest(token_id=token_id), self.assertRaisesRegex(
                ValueError, "ground truth.*special token"
            ):
                canonicalize_target_ids([3, token_id, 2], I2W)

    def test_unknown_token_id_reports_the_id(self):
        with self.assertRaisesRegex(KeyError, "99"):
            canonicalize_target_ids([3, 99, 2], I2W)

    def test_prediction_preserves_special_tokens_inside_content(self):
        stream = canonicalize_prediction_ids([1, 3, 0, 1, 2], I2W, maxlen=5)

        self.assertEqual(stream.tokens, ("note", "<pad>", "<bos>"))
        self.assertTrue(stream.terminated_by_eos)
        self.assertFalse(stream.truncated)

    def test_prediction_without_eos_at_maxlen_is_truncated(self):
        stream = canonicalize_prediction_ids([1, 3, 4, 10], I2W, maxlen=4)

        self.assertEqual(stream.tokens, ("note", "<s>", "c"))
        self.assertFalse(stream.terminated_by_eos)
        self.assertTrue(stream.truncated)

    def test_prediction_without_eos_before_maxlen_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "before maxlen"):
            canonicalize_prediction_ids([1, 3], I2W, maxlen=4)


class MetricViewTests(unittest.TestCase):
    def test_all_views_come_from_the_same_canonical_stream(self):
        stream = CanonicalTokenStream(
            ("ab", "<s>", "c", "<t>", "d", "<b>"),
            terminated_by_eos=True,
            truncated=False,
        )

        self.assertEqual(canonical_text(stream), "ab c\td\n")
        views = metric_views(stream)
        self.assertEqual(views.cer, tuple("ab c\td\n"))
        self.assertEqual(views.ser, ("ab", "c", "<t>", "d", "<b>"))
        self.assertEqual(views.ler, ("ab c\td",))

    def test_cer_uses_unicode_code_points_and_splits_multichar_tokens(self):
        stream = CanonicalTokenStream(("ab", "·"), True, False)

        self.assertEqual(metric_views(stream).cer, ("a", "b", "·"))

    def test_metrics_are_micro_averaged_over_the_split(self):
        predictions = (
            CanonicalTokenStream(("x",), True, False),
            CanonicalTokenStream(("b",), True, False),
        )
        targets = (
            CanonicalTokenStream(("a",), True, False),
            CanonicalTokenStream(("b", "c"), True, False),
        )

        cer, ser, ler = compute_canonical_metrics(predictions, targets)

        self.assertAlmostEqual(cer, 200.0 / 3.0)
        self.assertEqual(ser, 100.0)
        self.assertEqual(ler, 100.0)

    def test_empty_ground_truth_units_are_rejected(self):
        empty = (CanonicalTokenStream((), True, False),)

        with self.assertRaisesRegex(ValueError, "ground-truth CER"):
            compute_canonical_metrics(empty, empty)

    def test_prediction_and_target_counts_must_match(self):
        stream = CanonicalTokenStream(("note",), True, False)

        with self.assertRaisesRegex(ValueError, "same number"):
            compute_canonical_metrics((stream,), (stream, stream))

    def test_native_editdistance_matches_legacy_definition(self):
        alphabet = ("a", "b")
        sequences = [tuple(items) for length in range(4) for items in itertools.product(alphabet, repeat=length)]

        for left, right in itertools.product(sequences, repeat=2):
            with self.subTest(left=left, right=right):
                self.assertEqual(editdistance.eval(left, right), levenshtein_legacy(left, right))


if __name__ == "__main__":
    unittest.main()
