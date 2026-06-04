import feature_101night_analy as analy

import sys
import gc
import time
import argparse
from pathlib import Path

mff_path = "./Nathalie-40_20171011_121248.mff"

SleepEEGFeatureExtractor = analy.SleepEEGFeatureExtractor

ext = SleepEEGFeatureExtractor(
            mff_path, eeg_channel='E21',
            load_all_channels=False,
        )