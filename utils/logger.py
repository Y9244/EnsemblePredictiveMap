import joblib, os

class Logger:
    def __init__(self):
        self.sc_recordings = []
        self.pm_recordings = []

        self.base_path = '/Users/yk9244/thesis/src/Main/sparse_predictive_map/data'

    def append(self, s_h, p_h):
        self.sc_recordings.append(s_h)
        self.pm_recordings.append(p_h)

    def dump(self, name=None):
        joblib.dump(self.sc_recordings, os.path.join(self.base_path, f'{name}_sc.pkl'))
        joblib.dump(self.pm_recordings, os.path.join(self.base_path, f'{name}_pm.pkl'))
