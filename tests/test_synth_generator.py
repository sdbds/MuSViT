import os
import inspect
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import verovio
from PIL import Image

from experiments.full_page_omr.Generator.SynthGenerator import (
    SyntheticScoreRenderError,
    VerovioGenerator,
)


class _RejectingToolkit:
    def loadData(self, _music_sequence):
        return False


class _NoisyToolkit:
    def loadData(self, _music_sequence):
        os.write(2, b"Error: Inconsistent rhythm analysis\n")
        return True


def _generator_without_dataset(max_render_attempts=3):
    generator = object.__new__(VerovioGenerator)
    generator.max_render_attempts = max_render_attempts
    generator.render_attempts = 0
    generator.render_failures = 0
    generator.tokenization_mode = "bekern"
    return generator


class VerovioGeneratorTests(unittest.TestCase):
    def test_tokenization_default_is_bekern(self):
        default = inspect.signature(VerovioGenerator).parameters["tokenization_mode"].default

        self.assertEqual(default, "bekern")

    def test_unknown_tokenization_is_rejected_before_dataset_loading(self):
        with patch.object(VerovioGenerator, "load_beats") as load_beats:
            with self.assertRaisesRegex(ValueError, "tokenization_mode"):
                VerovioGenerator("source", tokenization_mode="standard")

        load_beats.assert_not_called()

    def test_render_rejects_failed_humdrum_import(self):
        generator = _generator_without_dataset()
        generator.tk = _RejectingToolkit()

        with self.assertRaisesRegex(SyntheticScoreRenderError, "rejected the generated Humdrum"):
            generator.render("invalid score")

        self.assertEqual(generator.render_attempts, 1)
        self.assertEqual(generator.render_failures, 1)

    def test_render_rejects_native_rhythm_error_even_when_import_returns_true(self):
        generator = _generator_without_dataset()
        generator.tk = _NoisyToolkit()

        with self.assertRaisesRegex(SyntheticScoreRenderError, "Inconsistent rhythm analysis"):
            generator.render("rhythmically inconsistent score")

        self.assertEqual(generator.render_attempts, 1)
        self.assertEqual(generator.render_failures, 1)

    def test_real_verovio_rhythm_conflict_is_rejected(self):
        generator = _generator_without_dataset()
        generator.tk = verovio.toolkit()
        padding = "\n".join(f"!! {'x' * 80}" for _ in range(30))
        inconsistent_score = padding + "\n" + """**kern\t**kern
*M4/4\t*M4/4
2c\t2e
.\t4f
4d\t4g
*-\t*-
"""

        with self.assertRaisesRegex(SyntheticScoreRenderError, "Inconsistent rhythm analysis"):
            generator.render(inconsistent_score)

        self.assertEqual(generator.render_attempts, 1)
        self.assertEqual(generator.render_failures, 1)

    def test_single_system_generation_retries_invalid_score(self):
        generator = _generator_without_dataset()
        generator.beat_db = {"*M4/4": ["4c <t> 4e"]}
        valid_svg = '<svg xmlns="http://www.w3.org/2000/svg"><g class="grpSym"/></svg>'
        generator.render = Mock(
            side_effect=[SyntheticScoreRenderError("invalid rhythm"), valid_svg]
        )
        generator.convert_to_png = Mock(return_value=np.zeros((8, 8, 3), dtype=np.uint8))

        image, tokens = generator.generate_music_system_image()

        self.assertEqual(generator.render.call_count, 2)
        self.assertEqual(image.shape, (4, 4, 3))
        self.assertEqual(tokens, ["<bos>", "4c", "<t>", "4e", "<eos>"])

    def test_full_page_generation_retries_until_system_count_matches(self):
        generator = _generator_without_dataset()
        generator.beat_db = {"*M4/4": ["4c <t> 4e <b> *- <t> *-"]}
        generator.textures = [str(
            Path(__file__).parents[1]
            / "experiments"
            / "full_page_omr"
            / "Generator"
            / "paper_textures"
            / "1.jpg"
        )]
        no_system_svg = '<svg xmlns="http://www.w3.org/2000/svg"/>'
        valid_svg = '<svg xmlns="http://www.w3.org/2000/svg"><g class="grpSym"/></svg>'
        generator.render = Mock(
            side_effect=[SyntheticScoreRenderError("invalid rhythm"), no_system_svg, valid_svg]
        )
        generator.convert_to_png = Mock(return_value=np.full((8, 8, 3), 255, dtype=np.uint8))
        generator.inkify_image = Mock(side_effect=lambda sample: Image.fromarray(sample))

        image, tokens = generator.generate_full_page_score(max_systems=1, strict_systems=True)

        self.assertEqual(generator.render.call_count, 3)
        self.assertEqual(image.shape, (4, 4, 3))
        self.assertEqual(tokens[0], "<bos>")
        self.assertEqual(tokens[-1], "<eos>")

    def test_generation_stops_after_bounded_attempts(self):
        generator = _generator_without_dataset(max_render_attempts=2)
        generator.beat_db = {"*M4/4": ["4c <t> 4e"]}
        generator.render = Mock(side_effect=SyntheticScoreRenderError("invalid rhythm"))

        with self.assertRaisesRegex(RuntimeError, "after 2 attempts"):
            generator.generate_music_system_image()

        self.assertEqual(generator.render.call_count, 2)


if __name__ == "__main__":
    unittest.main()
