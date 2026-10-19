# land_cover_tools

Build a 1 m × 1 m land-cover map from NAIP aerial imagery (Microsoft Planetary
Computer) with a U-Net trained on polygons you label yourself. The output is a
GeoTIFF where each pixel is an unsigned integer class ID.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

## Workflow

1. **Configure** `configs/default.yaml` (or a copy of it):
   - `imagery.bbox`: area of interest as `[min_lon, min_lat, max_lon, max_lat]`.
     NAIP covers the contiguous US only.
   - `classes`: your land-cover categories. Each key is the integer written to
     the output (1–65535; 0 is reserved for unlabeled/no data).

2. **Download imagery**

   ```bash
   python -m src.main download --config configs/default.yaml
   ```

   This searches NAIP for the bbox, mosaics the most recent year, and resamples
   it to an exact 1 m UTM grid in `data/imagery.tif` (4 bands: R, G, B, NIR).

3. **Label training areas in QGIS**: open `data/imagery.tif` and draw
   polygons into `data/labels.gpkg` with an integer `class_id` field. See
   [Working in QGIS](#working-in-qgis) for step-by-step instructions.

4. **Train and predict**

   ```bash
   python -m src.main train-predict --config configs/default.yaml
   ```

   Training cuts 256 px chips around your labels and fine-tunes a U-Net with
   an ImageNet-pretrained ResNet-34 encoder (weights download on first run).
   The best checkpoint goes to `data/model.pt`. Prediction tiles the whole
   image and writes:
   - `data/landcover.tif`: uint8 (uint16 if any ID > 255) class IDs on the same
     1 m grid as the imagery, nodata 0, with a color table if every class has
     a `color`.
   - `data/landcover.csv`: legend of class IDs, names and colors.

   `train` and `predict` can also be run separately, e.g. to re-predict after
   editing `prediction` settings.

5. **Check the map in QGIS**: see
   [Viewing the land-cover map](#viewing-the-land-cover-map).

## Working in QGIS

These steps were written for QGIS 3.x (free from <https://qgis.org>). Menu
names may differ slightly between versions.

### Opening the imagery

1. Start QGIS and create a new project (*Project → New*).
2. Add the image with *Layer → Add Layer → Add Raster Layer…*, choose
   `data/imagery.tif` and click *Add*. You can also drag the file from Finder
   or Explorer onto the map, or double-click it in the *Browser* panel.
3. The project takes the image's CRS, a UTM zone such as `EPSG:32610`, shown
   at the bottom right of the window. Keep it. Labels drawn in this CRS line
   up exactly with the 1 m pixels.
4. QGIS usually shows bands 1/2/3 as red/green/blue. If the colors look
   wrong or washed out, open the layer's *Properties → Symbology*:
   - *Render type*: **Multiband color**
   - *Red band* = Band 1, *Green band* = Band 2, *Blue band* = Band 3
   - *Min / Max Value Settings*: **Cumulative count cut** at 2–98%, then *Apply*
5. A false-color view often separates classes more clearly. Set *Red* =
   Band 4 (near-infrared), *Green* = Band 1 and *Blue* = Band 2. Vegetation
   turns bright red, water turns very dark, and pavement turns gray or cyan.
   Duplicate the layer (right-click → *Duplicate Layer*) to switch between
   the two views.
6. Pixels outside the imagery have the value 0, which is the file's nodata
   value, so QGIS draws them transparent.
7. Optional: for context, add a basemap with *Browser → XYZ Tiles →
   OpenStreetMap*, and drag it below the image in the *Layers* panel.

To check a pixel's values, select the *Identify Features* tool (the "i"
cursor icon) and click the image.

### Drawing the training labels

1. *Layer → Create Layer → New GeoPackage Layer…*
   - *Database*: `data/labels.gpkg` (the path set in `labels.path` in the config)
   - *Table name*: `labels`. Keep this the only layer in the file, because
     training reads the file's first layer.
   - *Geometry type*: **Polygon**
   - *CRS*: the image's CRS, e.g. `EPSG:32610`. Pick it from *Project CRS*.
   - Under *New Field*, add `class_id` with type **Whole Number (integer)**,
     then click *Add to Fields List*, then *OK*.
2. With the `labels` layer selected, turn on editing (the pencil icon or
   *Layer → Toggle Editing*), then choose *Add Polygon Feature*
   (Ctrl/Cmd + .).
3. Click to place the corners of a polygon and right-click to finish it. In
   the form that appears, enter the `class_id`, which is the class's number
   under `classes:` in the config (for example 3 for `tree_canopy`).
4. Click *Save Layer Edits* (the disk icon next to the pencil) regularly, and
   turn editing off when you're done. Training reads the saved file.

Tips for labeling:

- Pixels inside no polygon are ignored in training, so you don't need to
  label everything. Do label each class in several places across the image,
  and include edges between classes.
- Polygons don't need to be large. Many small, accurate polygons are better
  than a few large ones that cross into other classes.
- If polygons overlap, the one drawn later wins.
- To see classes while you draw, color the layer by class: *Properties →
  Symbology → Categorized*, *Value* = `class_id`, then *Classify*.
- To snap the edges of neighboring polygons together, use *Project →
  Snapping Options*. This avoids thin unlabeled gaps.

### Viewing the land-cover map

1. Add `data/landcover.tif` with *Layer → Add Layer → Add Raster Layer…*
   and drag it above the imagery in the *Layers* panel.
2. If every class in the config has a `color`, the file contains a color
   table, and QGIS shows it as **Paletted/Unique values** with those colors.
   Otherwise, set *Properties → Symbology → Render type* to **Paletted/Unique
   values** and click *Classify*.
3. The palette lists class IDs only. To show class names in the legend,
   double-click each entry in the *Label* column under *Symbology* and type
   the name from `data/landcover.csv`.
4. To compare the map with the imagery, set *Properties → Transparency →
   Global opacity* to about 50%, or toggle the layer's checkbox on and off.
5. With the *Identify Features* tool, clicking a pixel shows its class ID.
   Each pixel is 1 m × 1 m.
6. For the area of each class, run *Processing → Toolbox → Raster analysis →
   Raster layer unique values report* on `landcover.tif`. Its pixel counts
   equal square meters.

To improve the map, add or fix polygons in `labels.gpkg` where it's wrong,
then run `train-predict` again. QGIS reloads the updated `landcover.tif`
when you refresh the map (F5). If the map doesn't change, remove the layer
and add it again.

## Tips

- Label hard cases (shadows, dark roofs vs. water, dry grass vs. bare soil)
  and retrain; iterating on labels helps more than tuning hyperparameters.
- Every pixel gets one of the configured classes. Decide where surfaces
  without their own class belong (for example parking lots, driveways and
  sidewalks under `roads`) and label them consistently.
- Trees often hang over roads and roofs. Label what the camera sees from
  above: canopy over a road is `tree_canopy`.
- Training chips overlap (`chip_stride` < `chip_size`), so validation chips
  can share pixels with training chips and validation accuracy reads high.
  Check the output map visually.
- The whole image is held in memory, at roughly 50 bytes per m² during
  training and prediction (about 500 MB per 10 km²).

## Development

```bash
pytest -q
ruff check . && ruff format .
```
