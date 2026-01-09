"""Value residual helpers (ResFormer) for attention-only ablations."""


class ValueResidualState:
    def __init__(self):
        self.v1 = None

    def reset(self):
        self.v1 = None

    def mix(self, v, lamb1, lamb2):
        if self.v1 is None:
            self.v1 = v
        return lamb1 * v + lamb2 * self.v1.view_as(v)
