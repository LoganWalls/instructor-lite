from typing import List, Sequence

import pytest
import torch.nn.functional as F
from InstructorEmbedding import INSTRUCTOR
from numpy.testing import assert_equal

from instructor import InstructorEncoding, InstructorModel, ModelName


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


@pytest.fixture
def encoded_inputs(
    inputs: List[Sequence[str]], new_impl: InstructorModel
) -> InstructorEncoding:
    return new_impl.encode(inputs)


def test_token_embeddings(
    inputs: List[Sequence[str]],
    original_impl: INSTRUCTOR,
    new_impl: InstructorModel,
    encoded_inputs: InstructorEncoding,
):
    original_outputs = original_impl.encode(
        inputs,  # type: ignore
        output_value="token_embeddings",
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    _, new_token_embs = new_impl(encoded_inputs, normalize=False)
    for original, new in zip(original_outputs, new_token_embs):
        # Original implementation removes padding
        assert_equal(new[: original.shape[0]].numpy(), original.numpy())


def test_normalized_token_embeddings(
    inputs: List[Sequence[str]],
    original_impl: INSTRUCTOR,
    new_impl: InstructorModel,
    encoded_inputs: InstructorEncoding,
):
    original_outputs = original_impl.encode(
        inputs,  # type: ignore
        output_value="token_embeddings",
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    _, new_token_embs = new_impl(encoded_inputs, normalize=True)
    for original, new in zip(original_outputs, new_token_embs):
        # NOTE: original implementation does not normalize token embeddings,
        # even if `normalize_embeddings` is passed. So we must do it manually here.
        original = F.normalize(original, p=2, dim=1)
        # Original implementation removes padding
        assert_equal(new[: original.shape[0]].numpy(), original.numpy())


def test_full_text_embeddings(
    inputs: List[Sequence[str]],
    original_impl: INSTRUCTOR,
    new_impl: InstructorModel,
    encoded_inputs: InstructorEncoding,
):
    original_output = original_impl.encode(
        inputs,  # type: ignore
        output_value="sentence_embedding",
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    # (InstructorEmbeddings library always normalizes the sentence embeddings
    #  regardless of the parameters passed) so we must normalize the new model's outputs too.
    new_output, _ = new_impl(encoded_inputs, normalize=True)
    assert_equal(new_output.numpy(), original_output)


def test_index_conversion(
    inputs: List[Sequence[str]],
    encoded_inputs: InstructorEncoding,
    new_impl: InstructorModel,
):
    for i, (instr, text) in enumerate(inputs):
        ids = encoded_inputs.input_ids[i]
        text_start = encoded_inputs.text_char_to_token(i, 0)
        assert (
            new_impl.tokenizer.decode(ids[:text_start], skip_special_tokens=True)
            == instr
        )
        assert (
            new_impl.tokenizer.decode(ids[text_start:], skip_special_tokens=True)
            == text
        )
