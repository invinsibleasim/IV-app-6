
from damp_t.io_utils import load_light_iv, load_dark_iv, load_sunsvoc
from damp_t.translate import damp_t_pipeline
from damp_t.plotting import plot_iv_sets
import matplotlib.pyplot as plt

light = load_light_iv('examples/light_iv.csv')
dark  = load_dark_iv('examples/dark_iv.csv')
suns  = load_sunsvoc('examples/sunsvoc.csv')  # optional but recommended

res = damp_t_pipeline(light, dark, suns, target_G=1000.0, target_T=25.0)
fig = plot_iv_sets(res['anchor_curve'], res['translated'], res['neutralized'])
plt.show()

