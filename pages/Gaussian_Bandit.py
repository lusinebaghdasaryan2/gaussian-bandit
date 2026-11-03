# pages/Gaussian_Bandit.py

import pandas as pd
import streamlit as st

from gaussian_utils import (
    N_ALTERNATIVES,
    PRIOR_MEAN,
    PRIOR_VARIANCE,
    run_gaussian_thompson,
    validate_config,
)

st.title("Gaussian Bandit")
st.caption(
    "Thompson sampling for continuous rewards R | A=i ~ Normal(μᵢ, σ²) with known σ. "
    f"Every unknown mean has the same Normal({PRIOR_MEAN:g}, {PRIOR_VARIANCE:g}) prior."
)

with st.form("gaussian_bandit_form"):
    st.markdown("**True means** (used only by the simulator and for evaluation)")
    mean_cols = st.columns(N_ALTERNATIVES)
    defaults = [1.0, 2.0, 3.0]
    true_means = [
        col.number_input(f"True mean μ{i + 1}", value=defaults[i], step=0.5, format="%.2f", key=f"mu_{i}")
        for i, col in enumerate(mean_cols)
    ]

    c1, c2, c3 = st.columns(3)
    sigma = c1.number_input("Known σ (noise std)", value=1.0, step=0.5, format="%.2f")
    n_rounds = c2.number_input("Number of rounds", value=100, step=1, format="%d")
    seed = c3.number_input("Random seed", value=42, step=1, format="%d")

    run = st.form_submit_button("Run")

if run:
    errors = validate_config(true_means, sigma, n_rounds)
    if errors:
        for e in errors:
            st.error(e)
    else:
        # Session state keeps the latest result so it survives reruns and page switches.
        st.session_state.gaussian_result = run_gaussian_thompson(
            true_means, sigma, int(n_rounds), int(seed)
        )

result = st.session_state.get("gaussian_result")

if result is None:
    st.info("Set the parameters and press **Run**.")
else:
    st.subheader("Results")
    st.caption(
        f"True means {result.true_means} • σ = {result.sigma:g} • "
        f"{result.n_rounds} rounds • seed {result.seed}"
    )

    table = pd.DataFrame(
        {
            "Alternative": [f"Alternative {i + 1}" for i in range(N_ALTERNATIVES)],
            "True mean": result.true_means,
            "Posterior estimated mean": result.posterior_means,
            "Posterior std": [v**0.5 for v in result.posterior_variances],
            "Selection count": result.selection_counts,
        }
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Total observed reward", f"{result.total_reward:.2f}")
    m2.metric("Final recommendation", f"Alternative {result.recommendation + 1}")
    m3.metric("Selections (sum)", f"{sum(result.selection_counts)} / {result.n_rounds}")

    st.dataframe(
        table.style.format(
            {"True mean": "{:.3f}", "Posterior estimated mean": "{:.3f}", "Posterior std": "{:.3f}"}
        ),
        hide_index=True,
        width="stretch",
    )

    st.markdown(
        f"**Recommendation:** Alternative {result.recommendation + 1} "
        "(largest posterior mean)."
    )

    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("**Selection count by alternative**")
        st.bar_chart(table.set_index("Alternative")["Selection count"])
    with ch2:
        st.markdown("**Cumulative reward**")
        cumulative = pd.Series(result.rewards).cumsum()
        cumulative.index = range(1, len(cumulative) + 1)
        st.line_chart(cumulative)
