"""
MIT License

Copyright (c) 2024 Logan A. Walls

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from collections import OrderedDict
from typing import List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import BatchEncoding, T5EncoderModel, T5TokenizerFast
from transformers.utils import hub

ModelName = Literal[
    "hkunlp/instructor-base", "hkunlp/instructor-large", "hkunlp/instructor-xl"
]


class InstructorEncoding(BatchEncoding):
    def __init__(
        self,
        batch_encoding: BatchEncoding,
        instr_char_lengths: List[int],
        instr_token_lengths: List[int],
    ):
        # Copy over attrs from batch_encoding. Ideally we would
        # just call `super().__init__()`, but I can't find a clean
        # way to get T5Tokenizer to return this encoding class, so
        # this is a work-around.
        self.data = batch_encoding.data
        self._encodings = batch_encoding._encodings
        self._n_sequences = batch_encoding._n_sequences

        self.instr_char_lengths = instr_char_lengths
        self.instr_token_lengths = instr_token_lengths

    def text_char_to_token(self, batch_index: int, char_index: int) -> int:
        """
        Converts `char_index` from a text-relative character index to a
        instruction+text-relative token index suitable for indexing into the token
        embeddings produced by InstructorModel(encoding).

        For example: if `char_index` is 0, this function will return the token index
        that corresponds to the first token in the text, accounting for the
        fact that the instruction and text are concatenated.
        """
        return super().char_to_token(
            batch_index, self.instr_char_lengths[batch_index] + char_index
        )

    def to(self, device: torch.device) -> "InstructorEncoding":
        return InstructorEncoding(
            super().to(device),
            self.instr_char_lengths,
            self.instr_token_lengths,
        )


class InstructorModel:
    def __init__(self, model_name: ModelName, device: Optional[torch.device] = None):
        if device is None:
            device = torch.device("cpu")
        self.device = device
        self.tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained(model_name)
        self.hf_model: T5EncoderModel = T5EncoderModel.from_pretrained(model_name)  # type: ignore

        # Instructor models contain an additional layer that isn't part of the T5 architecture.
        # this loads that additional layer from the checkpoint.
        self.output_layer = torch.nn.Linear(
            self.hf_model.config.d_model,
            768,
            bias=False,  # type: ignore
        )
        output_weights_path: str = hub.cached_file(  # type: ignore
            model_name, "2_Dense/pytorch_model.bin"
        )
        state_dict = torch.load(output_weights_path, map_location=self.device)
        # Because the model was originally a sentence_transformers model,
        # these weights were nested inside a `sentence_transformers.Dense`
        # layer. But it does not use an activation function, so we can
        # load it as a `Linear` layer.
        state_dict = OrderedDict([("weight", state_dict["linear.weight"])])
        self.output_layer.load_state_dict(state_dict)

    def encode(self, pairs: List[Sequence[str]]) -> InstructorEncoding:
        batch_encoding = self.tokenizer(
            [instr + text for instr, text in pairs],
            padding=True,
            truncation="longest_first",
            return_tensors="pt",
        )
        instr_char_lengths = [len(instr) for instr, _ in pairs]
        instr_token_lengths = [
            batch_encoding.char_to_token(i, l) for i, l in enumerate(instr_char_lengths)
        ]

        return InstructorEncoding(
            batch_encoding, instr_char_lengths, instr_token_lengths
        )

    def __call__(
        self,
        encoding: InstructorEncoding,
        normalize: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """
        Processes instruction / text pairs and returns an embedding for each token in the text /
        instructions, as well as an aggregated embedding for the whole text.

        Args:
            encoding: encoded instruction / text pairs to be processed You can obtain an encoding from
                `InstructorModel.tokenize()`
            normalize: whether or not to normalize the embeddings

        Returns:
            a tuple with shape `(full_text_embs, token_embs)`
            where:
            `whole_text` is a `(len(pairs), emb_size)` tensor containing the aggregated embeddings
            `text_tokens` is a `(len(pairs), n_tokens(instr) + n_tokens(text), emb_size)` tensor containing the individual
                token embeddings for both the instructions and the texts
        """
        encoding = encoding.to(self.device)

        with torch.no_grad():
            hf_outputs = self.hf_model(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                output_hidden_states=True,
            )
            token_embs: Tensor = hf_outputs.last_hidden_state.detach()
            # For the full text embeddings, we don't want to include
            # information from the instructions so we modify the attention mask
            # to zero them out.
            attention_mask = encoding.attention_mask.clone()
            for i, instr_end in enumerate(encoding.instr_token_lengths):
                attention_mask[i].index_fill_(
                    0,
                    torch.arange(instr_end),
                    0.0,
                )

            # NOTE an unusual detail here: pooling is applied before
            # the output layer
            full_text_embs = self.output_layer(
                self.mean_pooling(
                    token_embs,
                    attention_mask,
                )
            )

        if normalize:
            token_embs = F.normalize(token_embs, p=2, dim=2)
            full_text_embs = F.normalize(full_text_embs, p=2, dim=1)

        return full_text_embs.detach(), token_embs.detach()

    @staticmethod
    def mean_pooling(token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Transformer models produce an embedding for each token. This function
        combines all of the tokens in a text to produce a single embedding
        for the whole text.
        """
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(  # type: ignore
            input_mask_expanded.sum(1), min=1e-9
        )
