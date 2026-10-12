"""Allow ``python -m cfp_compass run ...`` as a shortcut for ``python -m cfp_compass.pipeline``."""

from .pipeline import main

raise SystemExit(main())
