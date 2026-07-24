import os
import numpy as np
import torch

from collections import defaultdict
from pathlib import Path
from torch.utils.data import Dataset

class PDFDifficultyDataset(Dataset):
    def __init__(self, embeddings_path, split):
        self.pdfs = []
        data = defaultdict(list)

        for filename in os.listdir(embeddings_path):
            if filename.endswith('.npy'):
                # rsplit, so that sample ids containing underscores keep their id intact
                pdf_id, page, label = Path(filename).stem.rsplit('_', 2)

                if pdf_id not in split:
                    continue

                path = os.path.join(embeddings_path, filename)

                data[pdf_id].append((int(page), path, int(label)))

        for pdf_id, pages in data.items():
            pages = sorted(pages)

            paths = [p for _, p, _ in pages]
            labels = [l for _, _, l in pages]

            pdf_label = max(set(labels), key=labels.count)
            # Alternatives for pdf_label:
            #   Major label: max(labels)
            #   Mean label: sum(labels) / len(labels)
            self.pdfs.append((paths, pdf_label))

    def __len__(self):
        return len(self.pdfs)

    def __getitem__(self, idx):
        paths, label = self.pdfs[idx]

        # Each .npy already holds one mean-pooled vector per page, written by the encoder.
        embeddings = [torch.from_numpy(np.load(path)) for path in paths]

        pages = len(embeddings)

        return torch.stack(embeddings), pages, label

    def get_min_label(self):
        return min(label for _, label in self.pdfs)

    def get_max_label(self):
        return max(label for _, label in self.pdfs)

    def get_num_classes(self):
        return self.get_max_label() - self.get_min_label() + 1
