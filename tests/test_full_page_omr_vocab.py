import unittest
from unittest.mock import patch

from experiments.full_page_omr.utils import vocab_utils


class VocabularyFailureTests(unittest.TestCase):
    def test_broken_vocabulary_load_raises_the_source_exception(self):
        with (
            patch.object(vocab_utils.path, "isdir", return_value=True),
            patch.object(vocab_utils.path, "isfile", return_value=True),
            patch.object(vocab_utils.np, "load", side_effect=OSError("broken vocab")),
        ):
            with self.assertRaisesRegex(OSError, "broken vocab"):
                vocab_utils.check_and_retrieveVocabulary(
                    [],
                    "vocab",
                    "Polish_Scores_BeKern",
                )


if __name__ == "__main__":
    unittest.main()
