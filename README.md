# raymarch
A 3d renderer based on raymarching (WIP).
Renders planes and spheres for now.<br>
Requires cuda to run on GPU. Runs at ~0.02FPS on CPU whereas ~100FPS on GPU.

## To run
- Run `pip install -r requirements.txt`.
- Run `pygp.py` for now (beta).
- Run `python main.py` for the final version.
- `q` to quit.

## To do
- [x] fix shadow-cast issues
- [ ] python handle for subjects
- [ ] multiple light sources
- [ ] camera pose
  - [ ] stack system
  - [ ] abstract methods for transformations
- [ ] spherical mesh
- [ ] color blending
- [ ] soft shadows
  - [ ] surface emmitance
  - [ ] volumetric lighting
- [ ] lens blur
- [ ] reflections and refractions?
- [ ] physics simulator?
