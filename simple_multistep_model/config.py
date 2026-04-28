"""Feature flags for simple_multistep_model.

USE_RESIDUAL_BUCKETING: When True, the multistep pipeline routes through the
bucketed residual bootstrap path (per-location and per-(location, period)
residual pools). When False, the pipeline behaves exactly as on main.

This is a temporary toggle to keep the new code side-by-side with the old
during PR review; it should be removed once the bucketed path is the default.
"""

USE_RESIDUAL_BUCKETING: bool = False
