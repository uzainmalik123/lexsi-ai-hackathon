from ayasteer.gpu.capability import format_belebele, normalize_letter, score_exact_letter

ROW = {
    "flores_passage": "Passage text.",
    "question": "What is it?",
    "mc_answer1": "one",
    "mc_answer2": "two",
    "mc_answer3": "three",
    "mc_answer4": "four",
    "correct_answer_num": "3",
    "dialect": "urd_Arab",
    "question_number": "1",
    "link": "x",
}


def test_format_belebele_has_four_options_and_letter_target():
    prompt, target = format_belebele(ROW)
    assert target == "C"
    for line in ("A. one", "B. two", "C. three", "D. four"):
        assert line in prompt
    assert "Passage text." in prompt and "What is it?" in prompt
    assert prompt.rstrip().endswith("Answer with a single letter.")


def test_normalize_letter_takes_first_a_to_d():
    assert normalize_letter("C") == "C"
    assert normalize_letter("  c.") == "C"
    assert normalize_letter("The answer is B because") == "B"
    assert normalize_letter("جواب: D") == "D"
    assert normalize_letter("") == ""
    assert normalize_letter("none of these") == ""


def test_score_exact_letter():
    assert score_exact_letter("B", "Answer: B") == 1.0
    assert score_exact_letter("B", "A") == 0.0
