from codecs import ignore_errors
from mmdet.datasets import DATASETS, ConcatDataset, build_dataset


@DATASETS.register_module()
class PartialDataset(ConcatDataset):
    """Wrapper for full+partial supervision od."""

    def __init__(self, labeled: dict, partial: dict, **kwargs):
        ignore_keys = ["full", "unlabeled", "mixed"]
        for k in ignore_keys:
            if kwargs.get(k) is not None:
                kwargs.pop(k)
        super().__init__([build_dataset(labeled), build_dataset(partial)], **kwargs)

    @property
    def labeled(self):
        return self.datasets[0]

    @property
    def partial(self):
        return self.datasets[1]


@DATASETS.register_module()
class MixedDataset(ConcatDataset):
    """Wrapper for full+partial+unlabeled supervision od."""

    def __init__(self, labeled: dict, partial: dict, unlabeled: dict, **kwargs):
        ignore_keys = ["full", "mixed"]
        for k in ignore_keys:
            if kwargs.get(k) is not None:
                kwargs.pop(k)
        super().__init__([build_dataset(labeled), build_dataset(partial), build_dataset(unlabeled)], **kwargs)

    @property
    def labeled(self):
        return self.datasets[0]

    @property
    def partial(self):
        return self.datasets[1]

    @property
    def unlabeled(self):
        return self.datasets[2]


@DATASETS.register_module()
class SemiDataset(ConcatDataset):
    """Wrapper for mixed supervision od at cycle0."""

    def __init__(self, labeled: dict, unlabeled: dict, **kwargs):
        ignore_keys = ["full", "mixed", "partial"]
        for k in ignore_keys:
            if kwargs.get(k) is not None:
                kwargs.pop(k)
        super().__init__([build_dataset(labeled), build_dataset(unlabeled)], **kwargs)

    @property
    def labeled(self):
        return self.datasets[0]

    @property
    def unlabeled(self):
        return self.datasets[1]
