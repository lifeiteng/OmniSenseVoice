# modified from https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/conformer_ctc2/decode.py#L275

from typing import List, Tuple

import sentencepiece as spm
import torch


def ctc_greedy_search(
    ctc_probs: torch.Tensor,
    nnet_output_lens: torch.Tensor,
    sp: spm.SentencePieceProcessor,
    subsampling_factor: int = 4,
    frame_shift_ms: float = 10,
) -> Tuple[List[Tuple[float, float]], List[List[str]]]:
    """Apply CTC greedy search
    Args:
      ctc_probs (torch.Tensor):
        (batch, max_len, feat_dim)
      nnet_output_lens (torch.Tensor):
        (batch, )
      sp:
        The BPE model.
      subsampling_factor:
        The subsampling factor of the model.
      frame_shift_ms:
        Frame shift in milliseconds between two contiguous frames.

    Returns:
      utt_time_pairs:
        A list of pair list. utt_time_pairs[i] is a list of
        (start-time, end-time) pairs for each word in
        utterance-i.
      utt_words:
        A list of str list. utt_words[i] is a word list of utterence-i.
    """
    topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
    topk_index = topk_index.squeeze(2)  # (B, maxlen)
    mask = make_pad_mask(nnet_output_lens)
    topk_index = topk_index.masked_fill_(mask, 0)  # (B, maxlen)
    hyps = [hyp.tolist() for hyp in topk_index]

    start_token = b"\xe2\x96\x81".decode()  # '_'

    def get_first_tokens(tokens: List[str], blank_token: str) -> List[bool]:
        is_first_token = []
        first_tokens = []
        for t in range(len(tokens)):
            if tokens[t] != blank_token and (t == 0 or tokens[t - 1] != tokens[t]):
                is_first_token.append(True)
                first_tokens.append(tokens[t])
            elif t and tokens[t - 1] == tokens[t] and tokens[t].startswith(start_token):
                is_first_token.append(True)
                first_tokens.append(tokens[t])
            else:
                is_first_token.append(False)
        return first_tokens, is_first_token

    blank_token = sp.id_to_piece(0)

    utt_time_pairs = []
    utt_words = []
    for utt in range(len(hyps)):
        all_tokens = sp.id_to_piece(hyps[utt])
        first_tokens, is_first_token = get_first_tokens(all_tokens, blank_token=blank_token)
        index_pairs = parse_bpe_start_end_pairs(all_tokens, is_first_token, blank_token=blank_token)
        words = sp.decode(first_tokens).split()
        assert len(index_pairs) == len(words), (
            len(index_pairs),
            len(words),
            all_tokens,
        )
        start = convert_timestamp(
            frames=[i[0] for i in index_pairs],
            subsampling_factor=subsampling_factor,
            frame_shift_ms=frame_shift_ms,
        )
        end = convert_timestamp(
            # The duration in frames is (end_frame_index - start_frame_index + 1)
            frames=[i[1] + 1 for i in index_pairs],
            subsampling_factor=subsampling_factor,
            frame_shift_ms=frame_shift_ms,
        )
        utt_time_pairs.append(list(zip(start, end)))
        utt_words.append(words)

    return utt_time_pairs, utt_words


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths >= lengths.unsqueeze(-1)


def convert_timestamp(
    frames: List[int],
    subsampling_factor: int,
    frame_shift_ms: float = 10,
) -> List[float]:
    """Convert frame numbers to time (in seconds) given subsampling factor
    and frame shift (in milliseconds).

    Args:
      frames:
        A list of frame numbers after subsampling.
      subsampling_factor:
        The subsampling factor of the model.
      frame_shift_ms:
        Frame shift in milliseconds between two contiguous frames.
    Return:
      Return the time in seconds corresponding to each given frame.
    """
    frame_shift = frame_shift_ms / 1000.0
    time = []
    for f in frames:
        time.append(round(f * subsampling_factor * frame_shift, ndigits=3))

    return time


def parse_bpe_start_end_pairs(tokens: List[str], is_first_token: List[bool], blank_token: str) -> List[Tuple[int, int]]:
    """Parse pairs of start and end frame indexes for each word.

    Args:
      tokens:
        List of BPE tokens.
      is_first_token:
        List of bool values, which indicates whether it is the first token,
        i.e., not repeat or blank.

    Returns:
      List of (start-frame-index, end-frame-index) pairs for each word.
    """
    assert len(tokens) == len(is_first_token), (len(tokens), len(is_first_token))

    start_token = b"\xe2\x96\x81".decode()  # '_'
    # blank_token = "<blk>"

    non_blank_idx = [i for i in range(len(tokens)) if tokens[i] != blank_token]
    num_non_blank = len(non_blank_idx)

    pairs = []
    start = -1
    end = -1
    for j in range(num_non_blank):
        # The index in all frames
        i = non_blank_idx[j]

        found_start = False
        if is_first_token[i] and (j == 0 or tokens[i].startswith(start_token)):
            found_start = True
            if tokens[i] == start_token:
                if j == num_non_blank - 1:
                    # It is the last non-blank token
                    found_start = False
                elif is_first_token[non_blank_idx[j + 1]] and tokens[non_blank_idx[j + 1]].startswith(start_token):
                    # The next not-blank token is a first-token and also starts with start_token
                    found_start = False
        if found_start:
            start = i

        if start != -1:
            found_end = False
            if j == num_non_blank - 1:
                # It is the last non-blank token
                found_end = True
            elif is_first_token[non_blank_idx[j + 1]] and tokens[non_blank_idx[j + 1]].startswith(start_token):
                # The next not-blank token is a first-token and also starts with start_token
                found_end = True
            if found_end:
                end = i

        if start != -1 and end != -1:
            if not all([tokens[t] == start_token for t in range(start, end + 1)]):
                # except the case of all start_token
                pairs.append((start, end))
            # Reset start and end
            start = -1
            end = -1

    return pairs
