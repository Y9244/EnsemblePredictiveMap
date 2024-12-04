import pickle
from sparse_coding_1d import Sparse_Coding
import seaborn as sns
sns.set()

with open("bin/sc_1d_36.pkl", 'rb') as f:
    sparse_coding = pickle.load(f)

sparse_coding.calc_place_field()

sparse_coding.imshow_place_field(sparse_coding.place_field_recovered, 'sc_1d_36')


