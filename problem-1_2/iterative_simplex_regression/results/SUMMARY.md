# Tower index results summary

Confirmed index columns:
- col_11
- col_30
- col_42
- col_46
- col_48
- col_50

Confirmed/used exact decompositions:
- col_42 = 0.5955364658726565·col_26 + 0.40446096451682156·col_28
- col_50 = 0.586385·col_42 + 0.224753·col_32 + 0.188861·col_26
- col_11 = 0.342417·col_28 + 0.307571·col_42 + 0.212762·col_20 + 0.074628·col_07 + 0.062622·col_22
- col_48 = 0.5453628709082098·col_05 + 0.13482802176108735·col_45 + 0.12688200806246352·col_23 + 0.09683806773729109·col_04 + 0.09608857884372685·col_26
- col_30 = 0.2190·col_26 + 0.2140·col_19 + 0.1480·col_34 + 0.1460·col_40 + 0.1260·col_09 + 0.0790·col_45 + 0.0670·col_24
- col_46 = 0.2950·col_15 + 0.2370·col_34 + 0.1120·col_32 + 0.1220·col_09 + 0.0840·col_23 + 0.0800·col_05 + 0.0310·col_37 + 0.0260·col_20 + 0.0120·col_04

Classification used at submission:
- 6 indices
- 47 farmers (all remaining columns)

Key empirical separation used in discussion:
- Highest strict raw-validation RMSE among confirmed indices: ~0.2850 (`col_30`)
- Lowest RMSE among eventual farmer-labeled columns in final classification pass: ~1.4027 (`col_17`)
- Lowest RMSE among columns explicitly labeled `LIKELY FARMER`: ~2.1969 (`col_21`)

Pipeline stages represented by scripts:
1. Validate exact decompositions already known.
2. Run stability tests on known/candidate index supports.
3. Restrict to farmer-only predictor pool and rerun stepwise search.
4. Fill missing values from exact + approximate decompositions and rerun search.
5. Fill from all exact decompositions and rank all remaining columns.
6. Re-solve `col_30` and `col_46` with equality-constrained full-precision convex fits.
7. Classify the remaining columns against the known-farmer pool.
