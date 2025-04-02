from mlcg.nn.prior import Harmonic
from .utils import compute_raw_angles

class HarmonicRawAngle(Harmonic):
    name: Final[str] = "Angles"
    _order = 3

    def __init__(self, statistics) -> None:
        super(HarmonicRawAngle, self).__init__(statistics, HarmonicRawAngle.name)

    @staticmethod
    def neighbor_list(topology: Topology) -> dict:
        return Harmonic.neighbor_list(topology, HarmonicRawAngle.name)

    @staticmethod
    def compute_features(pos, mapping):
        return compute_raw_angles(pos, mapping)