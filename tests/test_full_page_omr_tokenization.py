from experiments.full_page_omr.tokenization import parse_kern_file


def test_pure_tokenization_module_owns_the_shared_bekern_parser():
    kern = "**kern\n*staff1\n*clefG2\n4c\n*-"

    assert parse_kern_file(kern, tokenization_mode="bekern") == ["4c", "<b>"]
