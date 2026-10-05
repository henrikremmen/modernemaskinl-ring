# LP-relaksasjon av vassdraget som modellinput — 28. september 2026

## Bunnlinje

Nytt forsøk: `notebooks/19_hydro_lp_relaxation.ipynb`. Vi løser én liten lineær relaksasjon (LP)
av hele Tokke–Vinje-vassdraget per case, bare fra tillatte inputs. LP-løsningen legges som
~12 features per stasjon til **nøyaktig** v5-modellen (public 0.99036). Alt annet holdes likt.
Lokale analyser uten ny boosting viser at LP-en fanger koblinger som den enkle prisregelen
mangler. Som tillegg til v5 bærer den også informasjon om ukentlig driftsnivå. Den løser ikke
nivåproblemet alene (Lio og Vinje G3 blir dårligere), og timingfeil innen uken gjenstår. Om dette gir
bedre score i fullmodellen er **ikke målt**. Du må kjøre notebooken, og den rapporterer selv
om historisk validering støtter kandidaten. Ingen score over 0.99562 kan loves.

## Hva som ble lest

Oppgave-, datasett- og innleveringstekstene. Topologi-YAML og rådata. Hele `src/`, `scripts/`
og `tests/`. Notebooks 01–18: markdown, kode og lagrede output. Eksperimentloggen, de tre
tidligere forskningsnotatene og lagrede prediksjoner/diagnoser under `artifacts/`. Det gjelder
også nb18-kjøringen (utvikling 2021, audit 2022, final).

## Hva som faktisk hjalp, og om sammenligningene var like

Fra public-scorene har feature-representasjon gitt de store hoppene: 0.82 → 0.974/0.981 → 0.988
→ fysikk 0.9899 → v4 0.99005 → v5 0.99036. Følgende hjalp ikke på public: modellmangfold uten
nye inputs (greedy OOF-blend 0.98774 mot XGB 0.98824, logistisk stack 0.98716), TCN (0.927),
ExtraTrees (0.945), EBM (0.789) og dispatch-forsøket i nb16 (0.98643). I nb16 var features og
kapasitet endret samtidig.

De lokale tallene er **ikke** sammenlignbare på tvers av notebooks:

- nb05/07 brukte ytre valideringsår til early stopping (optimistisk).
- nb13 brukte indre stopping.
- nb16–18 brukte faste, arvede trærantall.

Bare sammenligninger innen samme notebook er kontrollerte. Notebook 19 trener kontroll og
kandidat på identiske splitter, parametere, seeds og trærantall.

## Feilanalyse på lagrede v5-prediksjoner (2021 utvikling, 2022 audit)

- Micro-AUC var 0.98515 (2021) og 0.98682 (2022). Et label-informert orakel som kjenner hver
  case/enhets sanne ukentlige ON-andel gir 0.99740 og 0.99694. Dette er et **tak**, ikke et
  bevis på at timingen er riktig. Orakelet får alle konstante case/enheter gratis (46,7 % i
  2021, 39,0 % i 2022). Innen uker med både ON og OFF er v5s AUC 0.958/0.968 (vektet per
  positiv/negativ-par). Både nivå- og timingfeil betyr altså noe. (Orakelet er bare diagnose.)
- Svakeste enheter per enhets-AUC (2021/2022): Haukeli 0.945/0.946, Lio 0.966/0.963 og Vinje G3
  0.976/0.974. Ukenivå-MAE er høyest for Haukeli (0.167 i 2021) og Lio (0.116/0.113).
- Horisont: t0–t3 er dårligst (≈0.981), og siste dag er best (≈0.989). Månedene er ujevne:
  desember 2021 0.960, april 2022 0.963 og november 2022 0.968. UTC-time betyr lite
  (0.984–0.988).
- Den enkle regelen pris > sluttvannverdi i kildemagasinet (lokal EUR/MWh, jf. SHOP) gir
  følgende utrente AUC, snitt 2015–22:

  | Enheter | AUC |
  |---|---:|
  | Byrte | 0.989 |
  | Vesle Kjela | 0.984 |
  | Songa | 0.962 |
  | Kjela | 0.945 |
  | Tokke | 0.894–0.915 |
  | Hogga | 0.899 |
  | Lio | 0.897 |
  | Vinje | 0.77–0.79 |
  | Haukeli | 0.758 |

  Feilene sitter altså i koblede enheter, ikke i økonomien per stasjon.

## Pris- og vannverdifordeling

Prisnivået per case-startår var 9.3 i 2020, 75.9 i 2021, 211 i 2022, 78.9 i 2023 og 50.1 i
2024 (EUR/MWh). Lokale vannverdier fulgte med: Haukeli/Langeidvatn 372.6 i 2022 og 131.1 i 2023.
2022 er et ekstremt regime, og 2023/2024 ligger mellom 2021 og 2022. LP-ens andel av timer der
en stasjon «må» kjøre under egen vannverdi, er innenfor treningsspennet i testårene. Unntaket
er Lio i 2023 (0.50), som er rett over 2022 (0.48). Full tabell uten labels:
`artifacts/hydro_lp_v6/input_shift_by_year.csv`.

## Hvorfor Lio og Haukeli er vanskelige

**Lio** har minstevannføring i r_Vest_Vassdraget på 5–14 m³/s. Lio (maks 14 m³/s) er
billigste kilde når det naturlige tilsiget er for lavt. I timer der prisen er under
vannverdien, er Lio ON 43–93 % når kravet ikke dekkes, mot 1–28 % ellers (2015–22,
`hypothesis_audit/lio_min_flow.csv`). Signalet er en dual i en hydraulisk
restriksjon, ikke et pris-/vannverdisignal. LP-ens reduserte kost fanger det: 0.897 → 0.962.

**Haukeli** mates av Vatjern (0,3 Mm³, volumet er antatt fast 50 %) via et regulert elveløp fra
Langeidvatn. Enheten er liten (4,9 MW) og har en startkost på 143 EUR per start. LP-en løfter
utrent AUC fra 0.758 til 0.951. Start-/stoppøkonomien er ikke modellert, og det er trolig
restgapet.

## Hypotese

Med v5 som eneste kontroll: LP-features senker nivåfeilene for de koblede enhetene og øker
renset micro-AUC.

Støtte uten boosting:

- Utrent LP-margin har AUC Vinje G1 0.978, Tokke 0.984–0.987, Hogga 0.957 og Haukeli 0.951.
- **LP alene løser ikke nivåproblemet.** Ukenivå-MAE i 2022 er lavere enn v5 for bare 8 av 14
  enheter, og klart høyere for Lio (0.194 mot 0.114) og Vinje G3 (0.104 mot 0.051).
  (En tidligere versjon korrelerte fasit − v5 med LP − v5. Det er feil mål fordi begge deler
  v5-leddet, og det er erstattet.)
- **Som tillegg bærer LP nivåinformasjon utenfor utvalget.** En lineær nivåmodell tilpasset
  2021 og målt på 2022 er bedre med v5 + LP enn med bare v5 for 12 av 14 enheter (ikke Lio og
  Songa). Snitt-MAE går fra 0.0576 til 0.0490. Mot rå v5 (0.0511) er den bedre for 9 av 14.
  Se `hypothesis_audit/level_compare.csv`.
- En liten stacker ble tilpasset 2021 og målt på 2022. Kontrollen uten LP fikk 0.97821, og
  varianten med LP 0.98612. Begge ligger under v5 (0.98682), fordi ett års stacking forstyrrer
  kalibreringen mellom enheter. Per enhet slår LP-stackeren likevel v5: Tokke G1
  0.9948 → 0.9977 og Vinje G1 0.9864 → 0.9912.

Derfor er den reelle testen LP-features i fullmodellen, ikke stacking.

## Alternativer som ble vurdert og ikke valgt

- **Mer modellmangfold** (CatBoost med nye features, NN): tidligere mangfold ga ikke bedre
  public-score, og ingen lokal analyse peker på en feilkilde som mangfold alene løser.
- **Flere trær/features av samme type:** nb18 viste bare små gevinster.
- **Kalibrering/stacking:** feilet i tidligere prober og igjen her.
- **MILP med binære enheter per case:** tidsgrense og maskinhastighet gir risiko for resultater
  som ikke kan reproduseres. Vi valgte en deterministisk LP (HiGHS dual simplex, trådinvariant,
  testet). En start-kost-relaksasjon er et naturlig neste steg hvis LP støttes.

## LP-formulering (`src/sintef_hydro_lp.py`)

- Variabler per time: stasjonsvannføring (Songa fra to magasiner), forbitapping og flom med
  kostnader fra YAML (NOK|EUR per m³/s·t), regulerte elver med stykkevis kostnad, tunneler i
  begge retninger (tak 300 m³/s) og 17 magasinvolumer.
- Magasinbalanse i Mm³ (0,0036 per m³/s·t). r_Vest_Vassdraget er et knutepunkt til Bandak.
- Minstevannføring (6 elver) og minimumsvolum (3 magasiner) er myke med straff. SHOP bruker
  også myke volumkrav.
- Inntekt: pris × konvertering (MW per m³/s) i én enhets beste punkt ved startfallhøyden.
- Sluttverdi: SHOP-regelen (lokal EUR/MWh × konvertering i første stasjon under magasinet +
  global verdi nedstrøms). Kartet over «første stasjon» gjenskaper nøyaktig gruppene med like
  vannverdier i rådataene.
- Output per stasjon og time: vannføringsandel, dual-terskel, redusert-kost-margin og planlagte
  kilde-/nedstrømsvolumer. Features utledes fra disse, blant annet uke-/dagsandel,
  ±24 t-andeler og normaliserte marginer.
- Forenklinger: ingen p_min, startkost eller fallhøydeavhengighet, og tunnelene har fri
  overføring. Boosting-modellen skal korrigere for disse.
- Bruker ikke labels, SHOP-resultater, `total_calculation_time`, Run No eller observerte
  volumer etter case-start.

## Validering i notebook 19

- **Utvidende tidsvinduer med purge:** trening bare på case der hele 168-timershorisonten
  slutter før valideringsåret. Samme regel som `TimeSeriesSplit(gap=...)`, med gap lik
  horisontoverlappet.
- **Modellvalg på 2020 og 2021.** Regelen er forhåndsregistrert: snitt-delta for blend > 0 og
  intet år under −0,0005. Det er den eneste beslutningen. Ingen vekt- eller trærantallsøk;
  1×/1,5× lagres bare som diagnose.
- **Kontroll på 2022, kjørt én gang etterpå.** Ærlig: alle historiske år er brukt tidligere,
  og 2022 ble brukt i proben over. Usikkerheten rapporteres med en blokk-bootstrap med
  7-dagers blokker, siden nabocase deler 144 timer.
- **Logging:**
  - `results_log.csv`: AUC per år, modell og trærprefiks.
  - `runtime_log.csv`: kjøretid per enhet.
  - `diagnostics_*.csv`: enhet, startmåned, horisontblokk og UTC-time, med ukenivå-MAE.
  - `summary_*.json` og `predictions_*.npz`.
- Kontrollen sammenlignes automatisk med de lagrede v5-prediksjonene for 2022
  (reproduksjonssjekk).
- **Konklusjonen kommer før slutttreningen** og bruker bare resultater fra samme data, kode og
  parametere (`load_checked`). Kategoriene er:
  - `SUPPORTED`: valgt, og 90 %-intervallet for 2022 ligger over 0.
  - `POSITIVE BUT UNCERTAIN`: punktestimatet er positivt, men intervallet når 0.
  - `NOT CONFIRMED` og `NOT SUPPORTED`.

  Sluttcellen nekter å kjøre mot valideringer fra et annet oppsett. `final_manifest.json`
  knytter CSV-en til verdikten, og `TRAIN_FINAL = False` stopper før de 28 slutt-fitene.

## Nettkilder (primære)

- [SHOP vannverdier](https://docs.shop.sintef.energy/examples/water_values/water_values.html):
  `energy_value_input` er lokal og relativ til magasinet under første stasjon. Den konverteres
  i beste punkt ved startfallhøyde, og nedstrøms verdi legges til. Dette styrer sluttverdien i
  LP-en.
- [SHOP reservoar](https://docs.shop.sintef.energy/objects/reservoir/reservoir.html): minimumsvolum
  er en myk restriksjon med straff, og enheter er oppgitt.
- [SHOP elv](https://docs.shop.sintef.energy/objects/river/river.html): `flow_cost_const` og
  `flow_cost_curve` er i valuta per time per m³/s.
- [SINTEF InterOpt 2026](https://www.sintef.no/en/latest-news/2026/how-ai-could-squeeze-more-value-from-norwegian-hydropower/)
  og [prosjektsiden](https://www.sintef.no/en/projects/2024/interopt/): kjør først en rask
  relaksert modell, og bruk ML til å forutsi de nøyaktige commitment-beslutningene. Det er
  nettopp denne notebookens struktur.
- [Xavier, Qiu & Ahmed (INFORMS JoC 2021)](https://pubsonline.informs.org/doi/10.1287/ijoc.2020.0976):
  ML lærer fra tidligere løste UC-instanser som bare endrer seg litt fra dag til dag.
- [Gasse et al. (NeurIPS 2019)](https://arxiv.org/abs/1906.01629): variabelfeaturene inkluderer
  LP-løsningsverdi, redusert kost, dualverdier og om variabelen ligger på en grense (tabell
  lest i PDF-en). Det er presedens for å bruke LP-relaksasjonen som ML-input.
- [Pineda & Morales (2021)](https://arxiv.org/abs/2106.11687): enkle baselines må slås før man
  påstår gevinst av sofistikert læring. Derfor er kontrollen v5 og regelen forhåndsregistrert.
- [SciPy linprog/HiGHS](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html):
  marginals er sensitiviteter og har motsatt fortegn av vanlige Lagrange-multiplikatorer.
  `highs` velger algoritme selv, så vi låser `highs-ds`.
- [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html):
  trening alltid før test, og `gap` fjerner overlapp.

## Kodeendringer (bare nye filer; ingen eksisterende filer eller resultater er endret)

- `src/sintef_hydro_lp.py`: LP-en (statisk struktur, raske løsninger per case, trådparallell
  og deterministisk) og LP-features per stasjon.
- `src/sintef_lp_experiment.py`: kontroll/kandidat-runner med gjenopptakbare sjekkpunkter per
  enhet, gruppediagnoser, blokk-bootstrap, forhåndsregistrert valg, final og CSV-verifisering
  mot `sample_submission.csv`.
- `scripts/build_hydro_lp_notebook.py`: bygger notebook 19 og professor-notebooken med all kode
  innebygd.
- `scripts/audit_lp_hypothesis.py`: reproduserer tallene over fra lagrede nb18-prediksjoner
  (ca. 1,5 min, ingen base-trening). Resultatene ligger i
  `artifacts/hydro_lp_v6/hypothesis_audit/`.
- `tests/test_hydro_lp.py`: 10 nye tester.
- Genererte filer: `notebooks/19_hydro_lp_relaxation.ipynb`,
  `artifacts/hydro_lp_v6/Short_notebook_hydro_lp_v6.ipynb`, LP-cache,
  `physics_check_untrained.csv` og `input_shift_by_year.csv`.

## Tester og kontroller som faktisk er kjørt

- `pytest tests/`: **32 bestått**, 22 gamle og 10 nye. De nye dekker:
  - LP-økonomi: alt stengt når vannverdien er svært høy.
  - Kobling: Vinje begrenses av Vinjevatn/Tokke.
  - Dual-terskel lik vannverdi uten bindinger.
  - Minstevannføring som tvinger Lio til nøyaktig 10/14.
  - Ekte case er endelige og identiske med 1 og 2 tråder.
  - Feature-former.
  - Gjenopptak og isolasjon mellom variantene.
  - Beslutningsregelen.
  - Bootstrap og CSV-verifisering.
  - Toy-løp gjennom `evaluate_year` → `select_candidate` → `final_lp`.
  - Professor-notebooken, kjørt fra sine egne celler til en ekte CSV med falsk estimator.
  - Verdikt-kategoriene. Blant annet gir +0.00001 med intervall −0.0008..+0.0009 verdikten
    `POSITIVE BUT UNCERTAIN`.
  - Identitetsvakten og at konklusjonen står før slutttreningen.
- Notebook 19 sine celler uten trening er kjørt på ekte data. Bygget tok 56 s, med LP for
  3 647 case på 50 s ved 8 tråder. Kontrollen har 334 features, likt v5; kandidaten har
  346–347.
- Røyktest med ekte XGBoost/LightGBM-API: 1–2 trær, 30 case, alle 14 enheter, begge varianter.
  Maks RSS var 6,5 GB. Store modeller er **ikke** trent.
- LP-modulen er lik prototypen (maksavvik 3e-5) og trådinvariant (bit-identisk).

## Hva du kjører

1. Åpne `notebooks/19_hydro_lp_relaxation.ipynb`, se over første kodecelle (tråder og år), og
   kjør alle celler. Anslag ved 8 tråder: 2,5–3 timer for 14 sett med 14 enhetsmodeller
   (utvikling 8, kontroll 4, final 2). Avbrudd gjenopptas per enhet.
2. Les konklusjonscellen, `selection.json` og `summary_audit_2022.json`.
3. Les konklusjonen før slutttreningen. Hvis den sier «SUPPORTED», last opp
   `submissions/hydro_lp_v6.csv` som et nytt forsøk. «POSITIVE BUT UNCERTAIN» betyr at en
   opplasting er en test, ikke dokumentasjon. Ellers skal du ikke velge den. Behold v5 som valgt
   fallback i alle tilfeller. Enkeltmodellene (`xgb_/lgb_hydro_lp_v6.csv`) kan også lastes opp for
   diagnose.
4. For levering: `artifacts/hydro_lp_v6/Short_notebook_hydro_lp_v6.ipynb` er selvstendig og
   bruker 4 tråder, 28 fits og ingen cache. Anslått 1–2 timer. **Kjøretid og RAM er ikke målt
   kaldt på 4 kjerner**, så gjør det før innlevering.
5. Legg selv inn resultatet i `submissions/experiment_log.csv`.
