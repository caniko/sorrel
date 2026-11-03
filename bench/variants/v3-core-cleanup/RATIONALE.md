v2-core-cores with only step 3b changed, after the dev finding that the human uses two split patterns.

- Cores (a few small clean units carved from a noisy remainder) were handled by v2. Cleanups (human keeps the 70-90% majority as good and drops a contaminated minority) were not: the v2 core proposer took the minority as the core and the prompt then called the majority noise, inverting the human.
- The core proposer now also offers the cleanup reading (`split C by=core` returns core or cleanup; the handle line names it), so 3b acts on both words.
- Labels come from each part's own card, never from "the remainder is noise" by rule; a cleanup's kept majority is good only if its card is clean.
- A carved core in a root whose other parts are all noise-like needs a clean ACG and decent SNR before it is good (root-1 false positive: a clean-looking 9.2k carve out of an all-noise root).
- Everything else (steps 1-5, the rubric) is byte-identical to v2-core-cores.
