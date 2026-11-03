# DS223 Gaussian Bandit Frontend

Streamlit frontend with the original Beta–Bernoulli carousel bandit **plus** a new
**Gaussian Bandit** page (Thompson sampling for continuous rewards with known noise σ).

**Author(s):** _<your name(s)>_

## Project links

- GitHub repository: _<paste your repo URL>_
- Deployed Streamlit app: _<paste your Streamlit URL>_
- Starting application: https://carousel-mvp.streamlit.app/

## Credits

This project is based on the starting repository
[DS223_Group_Project](https://github.com/hovhannisyan91/DS223_Group_Project)
by hovhannisyan91 (now named `DS223_Bayesian_AB_Testing`). The source in this repository
comes from its `app/front/` folder; the Gaussian bandit page, `gaussian_utils.py` and the
README are my additions.

## Setup

Requires **Python 3.12** and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked
uv run streamlit run main.py
```

Then open http://localhost:8501 and use the sidebar to switch pages.

## Project structure

| File | Purpose |
|---|---|
| `main.py` | Entry point; registers all pages with `st.navigation` |
| `bandit_utils.py` | Original Beta–Bernoulli logic and session-state helpers |
| `pages/Create_Carousels.py`, `Interaction.py`, `Analytics.py` | Original pages (unchanged) |
| `gaussian_utils.py` | **New:** Gaussian Thompson sampling (validation, posterior update, simulator) |
| `pages/Gaussian_Bandit.py` | **New:** inputs, Run button and results display |

## Model 1: the original Beta–Bernoulli carousel bandit

**Alternative selection.** Each alternative has a click probability θ with a Beta(α, β)
posterior, starting at Beta(1, 1). `sample_carousel` draws one sample
`θ ~ Beta(α, β)` for every alternative (`random.betavariate`), sorts the samples from
highest to lowest, and shows the top `visible_count` alternatives. This is Thompson
sampling: alternatives with uncertain posteriors still get sampled high sometimes, so
they are explored, while good ones are shown more often.

**Reward recording.** On the Interaction page the user clicks one displayed alternative
or presses "No click / Next round". `update_after_feedback` appends an event to the
carousel's `history` with the displayed alternatives, the clicked one and the reward
(1 for a click, 0 for no click).

**Beta posterior update.** Every *displayed* alternative gets `impressions += 1`. The
clicked one gets `α += 1` and `clicks += 1`; each displayed alternative that was not
clicked gets `β += 1`. Alternatives that were not displayed are left unchanged.
Posterior mean = α / (α + β), and the "champion" is the alternative with the largest
posterior mean. A new sample is then drawn for the next round.

**Session state.** `st.session_state` is the shared memory between pages.
`init_state()` (called in `main.py` and on each page) creates `carousels` (a dict of
carousel id → alternatives table, history, round number, current display) and
`active_carousel_id`. Create Carousels writes into it, Interaction reads and updates
it, and Analytics reads it to draw posterior curves and reward history. The Gaussian
page stores its latest result in `st.session_state.gaussian_result`.

## Model 2: Gaussian Thompson sampling with known σ

Three alternatives. Choosing alternative *i* gives one continuous reward

R | A = i ~ Normal(μᵢ, σ²)

- σ > 0 is **known** and shared; the reward precision is 1/σ².
- Each μᵢ is **unknown** to the learner, with the same prior **Normal(0, 25)**
  (prior mean 0, variance 25, precision 1/25 = 0.04).

**Posterior update (conjugate).** After one reward r from alternative *i*:

- posterior precision: τ_new = τ_old + 1/σ²
- posterior mean: m_new = (τ_old · m_old + r/σ²) / τ_new

After n rewards this equals precision 1/25 + n/σ² and mean
(0/25 + Σr/σ²) / (1/25 + n/σ²): a precision-weighted average of the prior mean and the
data. Known σ is what makes the likelihood Gaussian in μ with fixed precision, which
keeps the posterior Normal.

**Each round:**
1. Sample a possible mean from every current Normal posterior.
2. Select the alternative with the largest sample.
3. The simulator draws one reward from Normal(μ_selected, σ²). Rewards stay continuous
   and can be negative.
4. Update only the selected alternative's posterior.

True means are used **only** to generate rewards and to evaluate results, never for
selection. The **final recommendation** is the alternative with the largest posterior
mean. Reproducibility comes from one `numpy.random.default_rng(seed)` generator.

## Experiments

Same prior, Normal(0, 25), and **same seed (42)** for all three runs. True means = [1, 2, 3].

| Experiment | σ | Rounds | Posterior means (A1, A2, A3) | Selection counts (A1, A2, A3) | Total reward | Recommendation |
|---|---|---|---|---|---|---|
| Short run | 1 | 100 | 1.570, 1.729, 2.995 | 4, 2, 94 | 291.52 | Alternative 3 |
| Longer run | 1 | 1,000 | 1.306, 2.142, 2.979 | 5, 13, 982 | 2960.20 | Alternative 3 |
| Higher noise | 3 | 1,000 | 1.154, 1.484, 2.954 | 21, 23, 956 | 2884.60 | Alternative 3 |

**Checks.** Re-running an identical configuration and seed gave identical rewards and
posteriors, and the selection counts summed to the number of rounds in every run
(100, 1,000, 1,000). Negative rewards occur (for example the minimum reward in the
higher-noise run was about −6.2).

### Discussion

**More observations.** With σ = 1, going from 100 to 1,000 rounds made the best
alternative's posterior tighter (std 0.10 → 0.03) and its estimate very close to the
truth (2.995 → 2.979 vs 3). Its selection share rose from 94% to 98.2%, and total
reward per round rose from 2.92 to 2.96, close to the best possible 3.

**Higher noise.** With σ = 3 and the same 1,000 rounds, the best alternative's
posterior std was about 0.10 instead of 0.03, and the other two alternatives were
chosen 44 times instead of 18. Noisier rewards make each observation less informative,
so posteriors stay wide longer and Thompson sampling keeps exploring. Total reward
fell from 2960 to 2885, which is the cost of that extra exploration.

**Unexpected results.** (1) The estimates for the *non-best* alternatives were not
accurate even with 1,000 rounds: in the σ = 1 run, A1 was still estimated at 1.31 (true 1)
and A2 at 2.14 (true 2), even though more rounds were available. This is expected for
Thompson sampling: once A3 looks best,
the others are rarely selected (5 and 13 times), so they receive few observations and
stay uncertain. The algorithm learns accurately where it matters for the decision,
not everywhere. (2) In the higher-noise run, A2 (true 2) was estimated at 1.48, below
its true value, and was selected only about as often as A1 (true 1), so the ordering of
the two worse alternatives was not reliably learned. Still, the final recommendation
(A3) was correct in all three runs.

As an extra robustness check (not part of the required table), I repeated each setting
with seeds 0–199: the recommendation was Alternative 3 in 200/200 runs for all three
settings, and the best alternative's average selection share was about 91.7%
(short), 98.8% (longer) and 93.3% (higher noise).

## Deployment (Streamlit Community Cloud)

Repository: this repo, branch `main`, entrypoint `main.py`, Python 3.12 (Advanced
settings). Dependencies are read from `pyproject.toml` / `uv.lock`.
