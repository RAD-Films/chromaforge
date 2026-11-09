## Performance

- **Camera profiles with tables develop on the graphics card again (process V9):** a photo whose camera profile has a hue/saturation map or a look table (a camera calibration, an imported `.dcp`, a DNG's own profile) applies those tables after the white balance since process V9, and that step ran on the processor, along with the rest of its stage, on every render. The graphics card now applies the tables itself, matching the processor to within 1 level in 8-bit output.
