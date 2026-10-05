# Researchgrunnlag for notebook 21 — 2026-09-29

## Utgangspunkt i prosjektet

- Notebook 19: LP-blend public 0.99571 (brukerens skjermbilde), senere-års-AUC 2022 0.992564.
- Notebook 20: ukentlig logit-skift med Ridge ga svakere total-AUC i begge år. Dette
  forsøket beholdt timerekkefølgen innen uken og testet derfor ikke timing direkte.
- Notebook 11 testet glatting og overlapp på et eldre ensemble. Ny glatting på v6 er
  en replikasjon på et nytt prediksjonsgrunnlag, ikke en nyoppfunnet metode.
- Notebook 14 prøvde plant states som full modell. Vi prøver nå en mye billigere
  etterkorreksjon, ikke en påstand om at plant states aldri har vært prøvd.
- Generatorene på Tokke/Vinje deler kraftverk og vann. Separate klassifikatorer kan
  ignorere avhengighet mellom generatorenes ON/OFF-valg.

## Primærkilder og hva de faktisk underbygger

1. [SINTEF SHOP: generator](https://docs.shop.sintef.energy/objects/generator/generator.html)
   beskriver minimumseffekt, binære commitment-valg og start-/stoppkostnader.
   Dette motiverer å teste tidsavhengighet. Vår dimensjonsløse kjedestraff er **ikke**
   en gjenskaping av SHOPs pengekostnader eller minimum opp-/nedetid.
2. [SINTEF SHOP: plant](https://docs.shop.sintef.energy/objects/plant/plant.html)
   beskriver kraftverksnivået og generatorfordeling. Dette motiverer en felles
   tilstandsmodell, men beviser ikke at en stasjonær frekvensprior gir bedre AUC.
3. [Stanford CS262: Conditional Random Fields, 2015](https://web.stanford.edu/class/cs262/archives/notes/lecture9.pdf)
   forklarer kjedemodeller og forward/backward. Vi bruker eksakt marginalisering
   over to tilstander, ikke Viterbi eller avrunding til harde 0/1-valg.
4. [Scikit-learn: VotingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)
   beskriver vektet gjennomsnitt av sannsynligheter. Vi tester bare en fast
   90/10 LP/v5-blend som reservehypotese, ikke en leaderboard-optimalisert vekt.
5. [Scikit-learn: probability calibration](https://scikit-learn.org/1.8/modules/calibration.html)
   forklarer at strengt monotone globale transformasjoner beholder rangering/AUC.
   Ren global temperaturjustering er derfor ikke prioritert som AUC-forbedring.

Eksterne kilder brukes kun som metodebakgrunn, aldri som ekstra treningsdata.

## Fem faste forsøk, ikke et stort søk

1. 3-timers logit-gjennomsnitt med svak blanding (0.15), som enkel timing-kontroll.
2. Soft kjede: sannsynlighet for en ON/OFF-sekvens er proporsjonal med produktet
   av modellscore og exp(-0.4 * antall skifter). Eksakte marginaler blandes inn med 0.25.
3. Svak prior for antall aktive generatorer på et kraftverk.
4. Matchet prior for hele generatorkombinasjonen, for å skille antall fra identitet.
5. 90 % original LP-blend + 10 % original v5-blend.

Plant-priorene estimeres kun fra purget treningsår. Empirisk felles fordeling
blandes med 75 % uavhengig fordeling, og korreksjonens log-vekt er 0.25.
Forholdet mellom felles og uavhengig fordeling brukes for å unngå å legge hele
treningsårets ON-rate oppå eksisterende modellprediksjoner. Alle tilstander er
mulige: ingen hard fysisk regel og ingen bytting av generator-ID-er.

## Evalueringsprotokoll og begrensninger

- 2020-prior → 2021-prediksjoner. Velg høyst én global kandidat på 2021.
- Lagre valget og konfigurasjonen før 2022 vises.
- 2021-prior → 2022-prediksjoner; fjern seks overlappende uker fra treningsåret.
- Vis også alle 2022-resultater som eksplorative diagnostikker. Ikke bytt kandidat
  etter å ha sett disse og kall samme år uavhengig validering.
- Micro-AUC er hovedmål. Vis Brier, ukentlig nivå-MAE, innen-uke-AUC og overgangsfeil.
- Færre skifter alene er ikke forbedring. Reelle kjøreplaner kan ha raske skifter.
- Ikke bland prediksjoner fra ulike ukecase selv om de gjelder samme klokkeslett:
  casene har forskjellig starttilstand og planleggingshorisont.
- Årene er allerede brukt i forskning. Fem kandidater medfører seleksjonsrisiko;
  positivt punktestimat er ikke signifikans eller dokumentasjon på privat score.
- Ingen ny submission, fulltrening, nettverkskall eller LP-løsning. Originalfiler
  leses bare. Egen unik outputmappe og før/etter-hasher av eksisterende CSV-er.
