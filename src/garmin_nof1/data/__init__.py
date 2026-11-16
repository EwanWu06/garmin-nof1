"""Data generators and loaders.

`synthetic` is public and committed — it produces a daily panel with a KNOWN
generative structure (mean-reverting HRV + sport-specific recovery cost) so the
validation scaffold and models can be tested before touching any real data.
Real-data loaders live alongside but read only from the gitignored `data/` tree.
"""
