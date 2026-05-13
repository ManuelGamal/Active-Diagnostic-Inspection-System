"""Minimal dataset stubs for RD++ pipeline compatibility."""


class MVTecDataset_test:
    """Stub - not used by active diagnostic pipeline."""
    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return None


def get_data_transforms(*args, **kwargs):
    """Stub - not used by active diagnostic pipeline."""
    return None, None
