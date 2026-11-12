### Add looks to an existing camera calibration
- `calibrate-lightroom fit --base FILE` keeps the camera colour a calibration already has and fits only the looks in your new exports on top of it. A look is added only when it beats the file's own colour on the photos you hold out (`--hold-out`).
- `--since-process N` adds the refit look for photos on that process version and newer only, so edits on older processes keep rendering as they did.
