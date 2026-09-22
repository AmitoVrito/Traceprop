# Paste this into a NEW cell in the same Colab session that just ran exp29
# --backend hf --n_subsets 500, IF that cell's variables are still alive
# (masks, margins, G_final, G_inline, G_test, dot_scores, trak_scores,
# lds_for -- all defined inside exp29's run() function, so this only works
# if you ran exp29 with `%run -i` or copied its body into the notebook
# directly rather than via `!python exp29_...py`, OR if you add this block
# to the bottom of exp29's run() before it returns).
#
# If the session is gone / you ran via `!python ...`, these variables never
# existed in the notebook's namespace -- skip this and instead rerun exp29
# (it now saves results/exp29_hf_gpt2_raw.npz automatically), then run
# exp32_paired_bootstrap.py on that file locally, no GPU needed.

import numpy as np

def lds_for_raw(attr, masks, margins, n_test):
    from scipy.stats import spearmanr
    pred = masks @ attr.T
    rs = np.array([spearmanr(pred[:, i], margins[:, i]).correlation for i in range(n_test)])
    return rs

r_final_dot = lds_for_raw(dot_scores(G_final, G_test), masks, margins, n_test)
r_inline_dot = lds_for_raw(dot_scores(G_inline, G_test), masks, margins, n_test)
r_final_trak = lds_for_raw(trak_scores(G_final, G_test), masks, margins, n_test)
r_inline_trak = lds_for_raw(trak_scores(G_inline, G_test), masks, margins, n_test)
rng2 = np.random.default_rng(0)
r_random = lds_for_raw(rng2.standard_normal((n_test, n_train)).astype(np.float32), masks, margins, n_test)

np.savez(
    "results/exp29_hf_gpt2_raw.npz",
    masks=masks, margins=margins,
    r_final_dot=r_final_dot, r_inline_dot=r_inline_dot,
    r_final_trak=r_final_trak, r_inline_trak=r_inline_trak,
    r_random=r_random,
)
print("saved results/exp29_hf_gpt2_raw.npz -- download this and run exp32_paired_bootstrap.py on it locally")
