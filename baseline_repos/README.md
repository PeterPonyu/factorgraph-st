# Baseline repository checkout instructions

This folder intentionally does **not** vendor third-party code. Run these commands from the FactorGraph-ST repository root when baseline execution begins.

```bash
mkdir -p .external_baselines
cd .external_baselines

git clone https://github.com/jiazhao97/INSPIRE.git INSPIRE
cd INSPIRE && git checkout 005d447374fea2820789d82936194690a40f69f0 && cd ..

git clone https://github.com/Seven595/HarveST.git HarveST
cd HarveST && git checkout 27ec12b303dd9a0ac8c9bc6accc99dd24fac48e2 && cd ..
```

Before code reuse, re-check licenses, repository HEADs, and paper/code terms. INSPIRE has `GitHub licenseInfo: null; no GitHub repository license inferred` as of 2026-05-27.
