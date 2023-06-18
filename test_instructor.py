from typing import List, Sequence

import pytest
import torch
import torch.nn.functional as F
from InstructorEmbedding import INSTRUCTOR
from numpy.testing import assert_equal

from instructor import InstructorModel, ModelName


@pytest.fixture
def model_name() -> ModelName:
    return "hkunlp/instructor-base"


@pytest.fixture
def original_impl(model_name: ModelName):
    return INSTRUCTOR(model_name)


@pytest.fixture
def new_impl(model_name: ModelName):
    return InstructorModel(model_name)


@pytest.fixture
def inputs():
    return [
        ["Represent the sentence", "This is a test."],
        ["Represent the phrase", "It is a different, (longer) test."],
        [
            "Represent the sentence for clustering",
            "This is a long long long test, with extra words.",
        ],
    ]


def test_token_embeddings(
    inputs: List[Sequence[str]],
    original_impl: INSTRUCTOR,
    new_impl: InstructorModel,
):
    original_outputs = original_impl.encode(
        inputs,  # type: ignore
        output_value="token_embeddings",
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    _, new_token_embs, new_instr_embs = new_impl(inputs, normalize=False)
    for original, new_instr, new_tok in zip(
        original_outputs, new_instr_embs, new_token_embs
    ):
        new = torch.cat([new_instr, new_tok], 0).numpy()
        assert_equal(new, original)


def test_normalized_token_embeddings(
    inputs: List[Sequence[str]],
    original_impl: INSTRUCTOR,
    new_impl: InstructorModel,
):
    original_outputs = original_impl.encode(
        inputs,  # type: ignore
        output_value="token_embeddings",
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    _, new_token_embs, new_instr_embs = new_impl(inputs, normalize=True)
    for original, new_instr, new_tok in zip(
        original_outputs, new_instr_embs, new_token_embs
    ):
        # NOTE: original implementation does not normalize token embeddings,
        # even if `normalize_embeddings` is passed. So we must do it manually here.
        original = F.normalize(original, p=2, dim=1)
        new = torch.cat([new_instr, new_tok], 0).numpy()
        assert_equal(new, original)


def test_full_text_embeddings(
    inputs: List[Sequence[str]],
    original_impl: INSTRUCTOR,
    new_impl: InstructorModel,
):
    original_output = original_impl.encode(
        inputs,  # type: ignore
        output_value="sentence_embedding",
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    # (InstructorEmbeddings library always normalizes the sentence embeddings
    #  regardless of the parameters passed) so we must normalize the new model's outputs too.
    new_output, _, _ = new_impl(inputs, normalize=True)
    assert_equal(new_output.numpy(), original_output)
