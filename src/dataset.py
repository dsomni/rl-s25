import re

import numpy as np


class RegexDataset:
    def __init__(
        self, texts: list[str], true_regex: str, shuffle: bool = True, seed: int = 420
    ) -> None:
        self._seed = seed
        self.shuffle = shuffle
        self._texts = texts

        self._targets = []
        self._word_counts = []
        for text in self._texts:
            target = [0 for _ in range(len(text))]
            words = 0

            for match in re.finditer(true_regex, text):
                words += 1
                start, end = match.span()
                for i in range(start, end):
                    target[i] = 1

            self._targets.append(target)
            self._word_counts.append(words)

        self._length = len(self._texts)

    def __len__(self):
        return self._length

    def create_iterator(self):
        inner_seed = self._seed
        while True:
            indices = list(range(self._length))
            if self.shuffle:
                np.random.seed(inner_seed)
                # np.random.shuffle(indices)
                inner_seed += 1
            for idx in indices:
                yield (
                    self._texts[idx],
                    self._targets[idx],
                    self._word_counts[idx],
                )

    # def __next__(self):
    #     item = (
    #         self._texts[self._idx],
    #         self._targets[self._idx],
    #         self._word_counts[self._idx],
    #     )

    #     self._idx += 1
    #     if self.idx >= self._length:
