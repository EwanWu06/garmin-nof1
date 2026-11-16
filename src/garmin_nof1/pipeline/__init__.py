"""Raw Garmin exports -> tidy, missingness-aware daily panel.

Modules here touch your private data and credentials. They read from / write to
the gitignored `data/` tree and are designed so the derived panel can always be
regenerated from locally archived raw FIT/JSON (resilient to Garmin auth changes).
"""
