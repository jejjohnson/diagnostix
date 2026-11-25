# Changelog

## [0.0.6](https://github.com/jejjohnson/diagnostix/compare/v0.0.5...v0.0.6) (2026-11-25)


### Bug Fixes

* **deps:** pipekit-array 0.1.6, gridax 0.3.2 and geophysix 0.0.8, without overrides ([#69](https://github.com/jejjohnson/diagnostix/issues/69)) ([74ec1bb](https://github.com/jejjohnson/diagnostix/commit/74ec1bb6be5f55d5c70facae83436ccd0f935b23))

## [0.0.5](https://github.com/jejjohnson/diagnostix/compare/v0.0.4...v0.0.5) (2026-11-25)


### Features

* **skill:** ensemble scores that a reliable ensemble of any size passes ([#62](https://github.com/jejjohnson/diagnostix/issues/62)) ([1724c6f](https://github.com/jejjohnson/diagnostix/commit/1724c6f68bcd7bb529d784387a3819a1ecc00318))
* **skill:** geostrophic imbalance, RMS divergence, density inversions and the QG balance residual ([#65](https://github.com/jejjohnson/diagnostix/issues/65)) ([50fa8b2](https://github.com/jejjohnson/diagnostix/commit/50fa8b23fdfb5a8957ce99d0fdf27ad36b153ea5))
* **skill:** pointwise scores, weighted, with gaps, and differentiable at a perfect prediction ([#61](https://github.com/jejjohnson/diagnostix/issues/61)) ([968687d](https://github.com/jejjohnson/diagnostix/commit/968687d33f90340b58d592699512969d64e2a203))
* **skill:** the Diebold–Mariano statistic of two methods' errors ([#66](https://github.com/jejjohnson/diagnostix/issues/66)) ([a0b7424](https://github.com/jejjohnson/diagnostix/commit/a0b74244dc20d1bd6a7681d2217bd8bc7d386c65))
* **skill:** the PSD score, the wavelength it resolves, and the RMSE in bands ([#64](https://github.com/jejjohnson/diagnostix/issues/64)) ([0b6c3c3](https://github.com/jejjohnson/diagnostix/commit/0b6c3c3073d6d66b63ed94add1495573fdf786f7))

## [0.0.4](https://github.com/jejjohnson/diagnostix/compare/v0.0.3...v0.0.4) (2026-11-25)


### Features

* **eofs:** area-weighted EOFs, projection and reconstruction ([#54](https://github.com/jejjohnson/diagnostix/issues/54)) ([cdf1424](https://github.com/jejjohnson/diagnostix/commit/cdf1424592e35d62e5574e1e0d813f59a2aac0a2))
* **spectra:** cross spectra, coherence with the average it needs, and rotary spectra ([#50](https://github.com/jejjohnson/diagnostix/issues/50)) ([102acfd](https://github.com/jejjohnson/diagnostix/commit/102acfdbca39e115818ce0f8000cc12176d557af))
* **spectra:** dealiased fluxes of kinetic energy and enstrophy, spectral slopes and scales ([#52](https://github.com/jejjohnson/diagnostix/issues/52)) ([9f8b3c9](https://github.com/jejjohnson/diagnostix/commit/9f8b3c922a6dd34d629ca999877a707f249fb0ef))
* **spectra:** power spectra in one and two dimensions, and the isotropic spectrum, that satisfy Parseval ([#49](https://github.com/jejjohnson/diagnostix/issues/49)) ([13e6858](https://github.com/jejjohnson/diagnostix/commit/13e6858c31a6b86aaa762996b1103136ca789832))
* **spectra:** the wavelet transform, its inverse and significance, and a local isotropic spectrum ([#53](https://github.com/jejjohnson/diagnostix/issues/53)) ([fafb8e5](https://github.com/jejjohnson/diagnostix/commit/fafb8e5210e477cb4c1935e19c0ff88d79cecdcf))

## [0.0.3](https://github.com/jejjohnson/diagnostix/compare/v0.0.2...v0.0.3) (2026-11-24)


### Features

* **budgets:** control volumes, the flux through their faces, and the flux divergence that closes with it ([#39](https://github.com/jejjohnson/diagnostix/issues/39)) ([e1433aa](https://github.com/jejjohnson/diagnostix/commit/e1433aaea6c8d501a612c71f1574c594f2da338a))
* **budgets:** the budget of a quantity term by term, with its residual, over a region and per cell ([#41](https://github.com/jejjohnson/diagnostix/issues/41)) ([dfd6419](https://github.com/jejjohnson/diagnostix/commit/dfd6419ea1eabde695678d4861a7157537441371))
* **climate:** anomalies and their inverse, linear trends and running means ([#43](https://github.com/jejjohnson/diagnostix/issues/43)) ([d703c93](https://github.com/jejjohnson/diagnostix/commit/d703c93ecadccf10858e40f3b48116a16fb20e5f))
* **climate:** daily and monthly climatologies on a leap-year calendar, with a base period and a harmonic fit ([#42](https://github.com/jejjohnson/diagnostix/issues/42)) ([f8b2192](https://github.com/jejjohnson/diagnostix/commit/f8b2192e7c3f73f6cb2f2105697f2f7c3849bc65))

## [0.0.2](https://github.com/jejjohnson/diagnostix/compare/v0.0.1...v0.0.2) (2026-11-24)


### Features

* **columns:** first crossings, and a mixed-layer depth that interpolates and is NaN on land ([#33](https://github.com/jejjohnson/diagnostix/issues/33)) ([9ad2d36](https://github.com/jejjohnson/diagnostix/commit/9ad2d3699478f00b538d7d144c0c6039b8b2b80b))
* **columns:** the bulk Richardson number and the boundary-layer height ([#34](https://github.com/jejjohnson/diagnostix/issues/34)) ([aa200cb](https://github.com/jejjohnson/diagnostix/commit/aa200cb10b09c18ae65f6687b397202b77bc3359))
* **columns:** the buoyancy frequency, the lapse rate and column integrals, on shared or per-column levels ([#36](https://github.com/jejjohnson/diagnostix/issues/36)) ([6fd3106](https://github.com/jejjohnson/diagnostix/commit/6fd3106d925e0d9f80cb99c0b7acf8375ca5eea8))
* **kinematics:** advection, frontogenesis, the ageostrophic velocity and the geostrophic residual ([#38](https://github.com/jejjohnson/diagnostix/issues/38)) ([3ee9262](https://github.com/jejjohnson/diagnostix/commit/3ee9262bca7e9236760da4fb26740ca481cf19cb))
* **kinematics:** the covariant velocity gradient on collocated grids, and shear and curvature vorticity ([#37](https://github.com/jejjohnson/diagnostix/issues/37)) ([9d3aa08](https://github.com/jejjohnson/diagnostix/commit/9d3aa080ed86bd3da921ab2a966e6383bf50ff05))

## 0.0.1 (2026-11-24)


### Features

* **reduce:** grouped sums over static groups, and masked, weighted reductions ([#29](https://github.com/jejjohnson/diagnostix/issues/29)) ([707c53e](https://github.com/jejjohnson/diagnostix/commit/707c53e8bee4c2284f4740ac091f0bf7ef6a10e7))


### Continuous Integration

* **release:** drop initial-version, which release-please ignores here ([#30](https://github.com/jejjohnson/diagnostix/issues/30)) ([4c3f033](https://github.com/jejjohnson/diagnostix/commit/4c3f0330df7ce8af13812da498cc0c66432da2db))

## Changelog
