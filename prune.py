"""
A script that takes a model as input and prunes it according to the given reduction args
"""

from harmonic_inference.data.data_types import TRIAD_REDUCTION
from harmonic_inference.utils.harmonic_utils import get_chord_label_list
from harmonic_inference.data.data_types import PitchType

get_chord_label_list(pitch_type=PitchType.TPC, use_inversions=False, reduction=TRIAD_REDUCTION)