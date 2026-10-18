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

3. **Label training areas in QGIS**
   - Open `data/imagery.tif`.
   - *Layer → Create Layer → New GeoPackage Layer…*: file `data/labels.gpkg`,
     geometry type Polygon, CRS = the image's CRS, and add an integer field
     `class_id`.
   - Draw polygons over representative areas of every class and set
     `class_id` to the class's key from the config. Pixels inside no polygon
     are ignored in training, so you don't need to label everything — but do
     label each class in several places across the image.

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

## Tips

- Label hard cases (shadows, dark roofs vs. water, dry grass vs. bare soil)
  and retrain; iterating on labels helps more than tuning hyperparameters.
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
