import numpy as np
import torch


def batch_preparation_img2seq(data):
    if len(data) != 1:
        raise ValueError(
            f"full-page OMR collate requires batch_size=1; received {len(data)} samples"
        )
    images = [sample[0] for sample in data]
    dec_in = [sample[1] for sample in data]
    gt = [sample[2] for sample in data]
    input_metadata = data[0][3] if len(data[0]) == 4 else None

    x_train = images[0]
    max_length_seq = max(len(sequence) for sequence in gt)
    decoder_input = torch.zeros(size=[len(dec_in), max_length_seq])
    target = torch.zeros(size=[len(gt), max_length_seq])

    for index, sequence in enumerate(dec_in):
        values = np.asarray([token for token in sequence[:-1]])
        decoder_input[index, 0 : len(sequence) - 1] = torch.from_numpy(values)

    for index, sequence in enumerate(gt):
        values = np.asarray([token for token in sequence[1:]])
        target[index, 0 : len(sequence) - 1] = torch.from_numpy(values)

    batch = (x_train, decoder_input.long(), target.long())
    return (*batch, input_metadata) if input_metadata is not None else batch
