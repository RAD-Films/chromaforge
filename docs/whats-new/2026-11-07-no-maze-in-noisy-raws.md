### No maze pattern in noisy raws
- Flat areas of high-ISO raws, such as a dim wall or a blurred background, could show short horizontal and vertical
  strokes instead of grain, often orange and blue under warm light. The demosaic chose between a horizontal and a
  vertical guess at each pixel, and in a flat area the noise alone decided which. Where neither guess is clearly
  better it now blends the two, so the noise looks like grain. Edges are as sharp as before. Thumbnails and previews
  of raws are redrawn.
