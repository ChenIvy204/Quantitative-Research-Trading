# Week 3 BSM Chooser Option Validation

This report uses the paper's Table 2 inputs as the fixed model parameters and simulates a two-stage geometric Brownian motion path. The simulation is stochastic, so row-by-row values are not expected to match the paper exactly:

### Formula

$$V_{chooser} = C(S, K, T) + P\left(S, K e^{-r(T-t)}, T-t\right)$$

$$S_t = S_0 * exp((mu - 0.5 * sigma^2) * t + sigma * sqrt(t) * Z)$$

### Replication Setup

- Random seed: 17170
- Price drift used in the GBM path: r - q = 0.0015 - 0.0233 = -0.0218
- Comparison rule: paper Table 3 rows are treated as a published reference, while the simulated rows come from the same Table 2 parameters under independent random draws

## Parameters

- Spot price: 156.7
- Strike price: 150.0
- Risk-free rate: 0.0015
- Dividend yield: 0.0233
- Volatility: 0.282
- Decision time: 0.5
- Maturity: 1.0
- Source note: Week 3 paper table: JPM stock price 156.7, risk-free rate 0.15%, sigma 28.2%, strike 150, dividend yield 2.33%, decision time 0.5 years, maturity 1 year

## Base Case

| metric                      | value      |
| --------------------------- | ---------- |
| spot_price                  | 156.700000 |
| strike_price                | 150        |
| adjusted_strike_for_put_leg | 149.887542 |
| risk_free_rate              | 0.001500   |
| dividend_yield              | 0.023300   |
| volatility                  | 0.282000   |
| decision_time_years         | 0.500000   |
| maturity_years              | 1          |
| time_to_decision_years      | 0.500000   |
| call_leg_value              | 18.688997  |
| put_leg_value               | 9.713217   |
| chooser_value               | 28.402215  |
| chooser_premium_vs_call     | 9.713217   |
| chooser_premium_vs_put      | 18.688997  |
| chooser_identity_gap        | 0          |

## Spot Sensitivity

| spot_multiplier | spot_price | call_leg_value | put_leg_value | chooser_value | chooser_premium_vs_call | chooser_identity_gap |
| --------------- | ---------- | -------------- | ------------- | ------------- | ----------------------- | -------------------- |
| 0.850000        | 133.195000 | 7.763969       | 22.484730     | 30.248699     | 22.484730               | 0                    |
| 0.950000        | 148.865000 | 14.470977      | 13.154096     | 27.625073     | 13.154096               | 0                    |
| 1               | 156.700000 | 18.688997      | 9.713217      | 28.402215     | 9.713217                | 0                    |
| 1.050000        | 164.535000 | 23.434825      | 7.017012      | 30.451837     | 7.017012                | 0                    |
| 1.150000        | 180.205000 | 34.302356      | 3.449730      | 37.752086     | 3.449730                | 0                    |

## Validation Notes

- The chooser value is the call leg plus the adjusted put leg.
- The identity gap should be numerically zero, up to floating-point rounding.
- The chooser premium over the call leg is always the value of the added put leg.

## Table 3 Simulation

The table below is generated from the Table 2 inputs. The exact path values depend on the seed and random draws, so they should be compared as a stochastic replication rather than a row-perfect match.

### Simulation Summary

| metric                       | value      |
| ---------------------------- | ---------- |
| rows_simulated               | 10         |
| spot_price                   | 156.700000 |
| strike_price                 | 150        |
| risk_free_rate               | 0.001500   |
| dividend_yield               | 0.023300   |
| risk_neutral_drift_r_minus_q | -0.021800  |
| volatility_from_current_data | 0.282000   |
| random_seed                  | 17170      |
| decision_time_years          | 0.500000   |
| maturity_years               | 1          |

### Simulated Paths

| row | z1        | z2        | simulated_st1 | choice_call_put | simulated_st2 | payoff    |
| --- | --------- | --------- | ------------- | --------------- | ------------- | --------- |
| 1   | 2.456421  | -0.973346 | 247.986098    | CALL            | 198.046618    | 48.046618 |
| 2   | -0.189113 | -0.149580 | 146.326778    | PUT             | 137.721638    | 12.278362 |
| 3   | 0.831693  | 1.003948  | 179.360048    | CALL            | 212.471007    | 62.471007 |
| 4   | -0.526638 | -0.204495 | 136.802515    | PUT             | 127.355215    | 22.644785 |
| 5   | -0.170727 | 0.145113  | 146.864221    | PUT             | 146.593513    | 3.406487  |
| 6   | 0.849695  | 1.644397  | 180.005043    | CALL            | 242.282243    | 92.282243 |
| 7   | -0.819602 | 1.162799  | 129.039707    | PUT             | 157.780688    | 0         |
| 8   | 1.350986  | -1.014785 | 198.928276    | CALL            | 157.560728    | 7.560728  |
| 9   | 1.558115  | 0.830742  | 207.316523    | CALL            | 237.251103    | 87.251103 |
| 10  | 0.126236  | -0.242662 | 155.823518    | CALL            | 143.962835    | 0         |

### Paper Reference Table

The next table shows the paper's published values only as a visual reference for comparison.

| row | paper_st1  | paper_choice | paper_st2  | paper_payoff |
| --- | ---------- | ------------ | ---------- | ------------ |
| 1   | 118.330000 | PUT          | 116.770000 | 33.230000    |
| 2   | 222.630000 | CALL         | 192.890000 | 42.890000    |
| 3   | 186.530000 | CALL         | 192.940000 | 42.940000    |
| 4   | 164.080000 | CALL         | 148.770000 | 0            |
| 5   | 159.090000 | CALL         | 116.780000 | 0            |
| 6   | 186.730000 | CALL         | 128.120000 | 0            |
| 7   | 106.610000 | PUT          | 90.520000  | 59.480000    |
| 8   | 163.060000 | CALL         | 179.610000 | 29.610000    |
| 9   | 129.260000 | PUT          | 144.820000 | 5.180000     |
| 10  | 115.410000 | PUT          | 136.500000 | 13.500000    |

### Aggregate Validation

This summary compares the simulated outputs against the paper at the distribution level. It includes the payoff mean, payoff standard deviation, payoff non-zero ratio, and the gap between the theoretical chooser value and the paper's average payoff.

| metric                        | paper     | simulated | difference |
| ----------------------------- | --------- | --------- | ---------- |
| payoff_mean                   | 22.683000 | 33.594133 | 10.911133  |
| payoff_std                    | 21.771920 | 36.188591 | 14.416670  |
| payoff_nonzero_ratio          | 0.700000  | 0.800000  | 0.100000   |
| theoretical_vs_paper_mean_gap | 28.402215 | 22.683000 | -5.719215  |
| call_count                    | 6         | 6         | 0          |
| put_count                     | 4         | 4         | 0          |

Interpretation: the paper's 10-row payoff sample is the reference distribution, while the simulated 10-row payoff sample comes from the same Table 2 parameters but different random draws. A non-zero gap is expected, and the statistics above show whether the two samples are in the same range.

### Side-by-Side Comparison: Prices

This split table keeps the price columns readable and shows the row-by-row differences directly.

| row | paper_st1  | simulated_st1 | st1_diff   | paper_st2  | simulated_st2 | st2_diff   |
| --- | ---------- | ------------- | ---------- | ---------- | ------------- | ---------- |
| 1   | 118.330000 | 247.986098    | 129.656098 | 116.770000 | 198.046618    | 81.276618  |
| 2   | 222.630000 | 146.326778    | -76.303222 | 192.890000 | 137.721638    | -55.168362 |
| 3   | 186.530000 | 179.360048    | -7.169952  | 192.940000 | 212.471007    | 19.531007  |
| 4   | 164.080000 | 136.802515    | -27.277485 | 148.770000 | 127.355215    | -21.414785 |
| 5   | 159.090000 | 146.864221    | -12.225779 | 116.780000 | 146.593513    | 29.813513  |
| 6   | 186.730000 | 180.005043    | -6.724957  | 128.120000 | 242.282243    | 114.162243 |
| 7   | 106.610000 | 129.039707    | 22.429707  | 90.520000  | 157.780688    | 67.260688  |
| 8   | 163.060000 | 198.928276    | 35.868276  | 179.610000 | 157.560728    | -22.049272 |
| 9   | 129.260000 | 207.316523    | 78.056523  | 144.820000 | 237.251103    | 92.431103  |
| 10  | 115.410000 | 155.823518    | 40.413518  | 136.500000 | 143.962835    | 7.462835   |

### Side-by-Side Comparison: Choice and Payoff

| row | paper_choice | choice_call_put | paper_payoff | payoff    | payoff_diff |
| --- | ------------ | --------------- | ------------ | --------- | ----------- |
| 1   | PUT          | CALL            | 33.230000    | 48.046618 | 14.816618   |
| 2   | CALL         | PUT             | 42.890000    | 12.278362 | -30.611638  |
| 3   | CALL         | CALL            | 42.940000    | 62.471007 | 19.531007   |
| 4   | CALL         | PUT             | 0            | 22.644785 | 22.644785   |
| 5   | CALL         | PUT             | 0            | 3.406487  | 3.406487    |
| 6   | CALL         | CALL            | 0            | 92.282243 | 92.282243   |
| 7   | PUT          | PUT             | 59.480000    | 0         | -59.480000  |
| 8   | CALL         | CALL            | 29.610000    | 7.560728  | -22.049272  |
| 9   | PUT          | CALL            | 5.180000     | 87.251103 | 82.071103   |
| 10  | PUT          | CALL            | 13.500000    | 0         | -13.500000  |

## Validation Conclusion

- The implementation now clearly distinguishes model inputs, simulation assumptions, and the paper's reference outputs.
- Exact path-level equality is not expected because the simulated Table 3 uses random draws, but the aggregate comparison makes the deviation visible.
- The most likely reasons for differences versus the paper are the random seed and the paper's own simulation settings, which are not fully specified in the table excerpt.
- The report now states the seed, the GBM update rule, and the drift assumption so the run is reproducible.