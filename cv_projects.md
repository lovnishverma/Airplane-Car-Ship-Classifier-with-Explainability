# Computer Vision Project Ideas
### MobileNetV2 Transfer Learning Pipeline — Adaptations & Datasets

> Base pipeline: MobileNetV2 + two-phase training + Grad-CAM + INT8 TFLite  
> All projects below reuse this pipeline. Only the **delta** from baseline is documented.

---

## How to Read This Document

Each project lists:
- **Classes** — what the model predicts
- **Dataset** — where to get training data (free unless noted)
- **Pipeline adjustments** — exact changes needed from the base airplane/car/ship code
- **GradCAM value** — why explainability matters for this specific use case
- **Deployment target** — who uses it and how
- **Difficulty** — dataset quality / problem complexity (not pipeline complexity)

---

## Domain 1 — Agriculture & Food

---

### 1.1 Crop Disease Detection ⭐ Top Pick

**Problem:** Classify leaf images to detect common crop diseases before they spread.

**Classes:**
- `healthy`
- `bacterial_blight`
- `fungal_infection`
- `pest_damage`
- `nutrient_deficiency`

**Dataset:**
- [PlantVillage Dataset — Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) — 54,000+ images, 38 classes, free
- Subset to 4–5 classes relevant to Punjab crops (wheat, rice, maize)

**Pipeline adjustments:**
```python
# 1. Augmentation — add color jitter (leaf color IS diagnostic)
layers.RandomHue(0.05),
layers.RandomSaturation(0.15),
layers.RandomBrightness(0.15),

# 2. Flip — both directions are fine for leaves
layers.RandomFlip("horizontal_and_vertical"),

# 3. Input size — increase to 224x224 for fine-grained texture detail
IMG_SIZE = (224, 224)

# 4. class_names update
class_names = ['bacterial_blight', 'fungal_infection', 'healthy',
               'nutrient_deficiency', 'pest_damage']
```

**No other changes needed.**

**GradCAM value:** Shows the exact lesion location on the leaf. A farmer or agricultural extension worker can visually confirm the affected area — critical for adoption.

**Deployment target:** Android app (TFLite) for farmers. Offline-capable — no internet needed in fields.

**Difficulty:** 🟢 Easy — large, clean, well-labelled dataset

---

### 1.2 Fruit Quality Grading

**Problem:** Grade fruits as export-quality, local-market, or reject at packing houses.

**Classes:**
- `grade_A` (export quality)
- `grade_B` (local market)
- `reject` (rotten / heavily bruised)

**Dataset:**
- [Fruit Disease Detection — Kaggle](https://www.kaggle.com/datasets/enalis/fruit-diseases) — apples, mangoes, bananas
- [Fruits 360 — Kaggle](https://www.kaggle.com/datasets/moltean/fruits) — 90,000+ images

**Pipeline adjustments:**
```python
# 1. Background removal matters here — fruits are often on white/black bg
# Add RandomBackground augmentation or use segmentation preprocessing

# 2. Color augmentation critical — ripeness is color-dependent
layers.RandomHue(0.08),
layers.RandomSaturation(0.2),

# 3. Flip both directions — fruit orientation doesn't matter
layers.RandomFlip("horizontal_and_vertical"),
layers.RandomRotation(0.5),  # full rotation — fruit on conveyor can be any angle

# 4. Consider regression head instead of 3-class softmax
# for continuous quality scoring (0.0 to 1.0)
```

**GradCAM value:** Highlights bruise, rot, or discoloration patches — packing house supervisor can audit rejections.

**Deployment target:** Raspberry Pi camera at packing conveyor belt. TFLite export is the deployment artifact.

**Difficulty:** 🟢 Easy

---

### 1.3 Soil Type Classification

**Problem:** Classify soil from field photos to recommend fertilizer and crop type.

**Classes:**
- `alluvial` (most of Punjab)
- `black_cotton`
- `red_laterite`
- `sandy`
- `clayey`

**Dataset:**
- [Soil Image Dataset — Kaggle](https://www.kaggle.com/datasets/prasanshasatpathy/soil-types) — ~1,500 images
- May need to augment with web scraping (small dataset)

**Pipeline adjustments:**
```python
# 1. Dataset is small — increase augmentation aggressiveness
layers.RandomZoom(0.3),         # wider zoom range
layers.RandomContrast(0.4),     # soil texture varies hugely with lighting
layers.RandomRotation(0.5),

# 2. Use MixUp more aggressively — alpha=0.4 instead of 0.2
# Soil classes have smooth perceptual boundaries

# 3. Unfreeze more backbone layers in Phase 2 (top 50 instead of 30)
# Texture classification needs deeper fine-tuning
for layer in base_model.layers[-50:]:
    layer.trainable = True

# 4. Increase Phase 1 epochs — smaller dataset needs more head warmup
PHASE1_EPOCHS = 25

# 5. EarlyStopping patience = 8 (more patience for small dataset)
```

**GradCAM value:** Shows which texture region drove classification — agronomist can verify it's reading soil texture not background grass/debris.

**Deployment target:** Government agriculture portal, state extension worker app.

**Difficulty:** 🟡 Medium — small dataset requires careful augmentation

---

### 1.4 Rice Grain Quality Classification

**Problem:** Classify rice grain samples as premium, standard, or broken/mixed for milling QC.

**Classes:**
- `premium` (whole, uniform)
- `standard`
- `broken`
- `chalky`
- `discolored`

**Dataset:**
- [Rice Image Dataset — Kaggle](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset) — 75,000 images, 5 varieties
- Build custom quality dataset with microscope camera

**Pipeline adjustments:**
```python
# 1. Input size increase — grain texture is fine-grained
IMG_SIZE = (224, 224)

# 2. No horizontal-only restriction — grain orientation is irrelevant
layers.RandomFlip("horizontal_and_vertical"),
layers.RandomRotation(0.5),  # grains appear at all angles

# 3. Macro photography preprocessing
# Grains are photographed on uniform backgrounds
# Add center crop augmentation
layers.CenterCrop(200, 200),  # remove edge artifacts from lightbox setup
```

**GradCAM value:** Highlights the specific grain defect region — QC operator can spot-check automated rejections.

**Deployment target:** Milling factory QC station. Webcam + laptop running Gradio locally.

**Difficulty:** 🟢 Easy

---

## Domain 2 — Healthcare & Medical

---

### 2.1 Eye Disease Screening ⭐ Top Pick

**Problem:** Screen fundus (retinal) images for common eye diseases at rural eye camps.

**Classes:**
- `normal`
- `diabetic_retinopathy`
- `glaucoma`
- `cataract`
- `age_related_macular_degeneration`

**Dataset:**
- [ODIR-5K — Kaggle](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k) — 5,000 patient records
- [Diabetic Retinopathy — Kaggle](https://www.kaggle.com/competitions/diabetic-retinopathy-detection)

**Pipeline adjustments:**
```python
# 1. Preprocessing — fundus images need circular crop and CLAHE enhancement
import cv2
def preprocess_fundus(img):
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

# 2. Input size — medical imaging needs higher resolution
IMG_SIZE = (224, 224)  # or 299x299 if using InceptionV3

# 3. Class imbalance handling — diseased images are fewer
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

# 4. Stricter augmentation — no extreme distortions for medical images
layers.RandomFlip("horizontal"),  # horizontal only
layers.RandomRotation(0.1),       # very small rotation only
layers.RandomZoom(0.05),          # minimal zoom

# 5. Consider switching backbone to EfficientNetB3 for medical imaging
base_model = tf.keras.applications.EfficientNetB3(
    include_top=False, weights='imagenet', input_shape=(224,224,3)
)

# 6. Threshold tuning after training — high recall preferred over precision
# Better to over-refer than miss a disease
```

**⚠️ Important framing:** Always deploy as a *triage/screening* tool, not diagnosis. Output must include confidence score and "refer to ophthalmologist if score > X" logic.

**GradCAM value:** Mandatory for clinical trust. Shows which retinal region (optic disc, macula) triggered prediction. Ophthalmologist can validate or override.

**Deployment target:** Eye camp tablet app. Rural health worker uses it before referring to specialist.

**Difficulty:** 🔴 Hard — class imbalance, image preprocessing, clinical validation required

---

### 2.2 Skin Condition Triage

**Problem:** Help rural health workers identify common skin conditions for referral decisions.

**Classes:**
- `normal`
- `fungal_infection`
- `eczema`
- `psoriasis`
- `acne_severe`

**Dataset:**
- [DermNet — Kaggle](https://www.kaggle.com/datasets/shubhamgoel27/dermnet) — 19,500 images, 23 classes
- [HAM10000 — Kaggle](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection)

**Pipeline adjustments:**
```python
# 1. Color calibration matters — skin photos taken under different lighting
layers.RandomHue(0.05),
layers.RandomBrightness(0.2),

# 2. Fitzpatrick scale awareness — model must work across skin tones
# Ensure training data has diverse skin tones — audit dataset before training

# 3. Class imbalance — use focal loss instead of categorical crossentropy
import tensorflow_addons as tfa
loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=0.25, gamma=2.0)

# 4. Test-time augmentation (TTA) for higher confidence
def predict_with_tta(model, img, n=10):
    preds = [model(augment(img)) for _ in range(n)]
    return np.mean(preds, axis=0)
```

**GradCAM value:** Shows affected skin region — helps health worker confirm the model is reading the lesion, not lighting artifacts or background clothing.

**Deployment target:** ASHA worker mobile app in rural health program.

**Difficulty:** 🔴 Hard — skin tone bias, lighting variation, regulatory framing

---

### 2.3 Yoga Pose Classification

**Problem:** Real-time pose correction app for yoga/fitness.

**Classes:**
- `tree_pose`
- `warrior_I`
- `warrior_II`
- `downward_dog`
- `cobra`
- `mountain_pose`
- `child_pose`

**Dataset:**
- [Yoga Poses Dataset — Kaggle](https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset) — 1,500 images
- [Yoga-82 Dataset](https://sites.google.com/view/yoga-82/home) — 28,000 images, 82 classes

**Pipeline adjustments:**
```python
# 1. Person segmentation preprocessing recommended
# Remove background — background should not influence pose prediction
# Use MediaPipe for keypoint extraction as preprocessing step

# 2. Aggressive spatial augmentation — poses appear at all scales
layers.RandomFlip("horizontal"),  # horizontal only — left/right matters
layers.RandomRotation(0.2),
layers.RandomZoom(0.25),

# 3. Consider keypoint-based approach instead of raw pixels
# MediaPipe Pose → 33 keypoints → lightweight classifier
# More robust than pixel-based for pose classification

# 4. Temporal smoothing for real-time video
from collections import deque
pred_buffer = deque(maxlen=5)
# Average last 5 frame predictions to reduce jitter
```

**GradCAM value:** Confirms model is looking at body position, not background gym equipment or clothing color.

**Deployment target:** Mobile app, webcam-based desktop app via Gradio.

**Difficulty:** 🟡 Medium

---

### 2.4 X-Ray Abnormality Detection

**Problem:** Screen chest X-rays for common findings at primary health centres.

**Classes:**
- `normal`
- `pneumonia`
- `pleural_effusion`
- `cardiomegaly`

**Dataset:**
- [Chest X-Ray Images — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — 5,856 images
- [NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/data) — 112,000 images

**Pipeline adjustments:**
```python
# 1. Grayscale to RGB conversion — X-rays are single channel
def xray_preprocess(img):
    if img.shape[-1] == 1:
        img = tf.image.grayscale_to_rgb(img)
    return img

# 2. CLAHE enhancement like fundus — critical for X-ray contrast
# Apply before feeding to model

# 3. No color augmentation — X-rays are grayscale
# REMOVE: RandomHue, RandomSaturation, RandomContrast
# KEEP: RandomFlip (horizontal only), RandomZoom, RandomRotation (very small)

# 4. Weighted loss — normal cases dominate most datasets
# Must balance or model predicts "normal" for everything

# 5. Backbone — consider DenseNet121 (standard for medical imaging)
base_model = tf.keras.applications.DenseNet121(
    include_top=False, weights='imagenet', input_shape=(224,224,3)
)
```

**GradCAM value:** Highlights lung region of abnormality — radiologist can verify or flag errors. Regulatory requirement in any clinical deployment.

**Deployment target:** Primary health centre tablet. Radiologist reviews flagged cases remotely.

**Difficulty:** 🔴 Hard — grayscale images, clinical validation, class imbalance

---

## Domain 3 — Industry & Manufacturing

---

### 3.1 Fabric Defect Detection ⭐ Top Pick

**Problem:** Automated quality control on textile production lines.

**Classes:**
- `good`
- `thread_pull`
- `hole`
- `stain`
- `weave_defect`
- `pilling`

**Dataset:**
- [AITEX Fabric Dataset — Kaggle](https://www.kaggle.com/datasets/nexuswho/aitex-fabric-image-database)
- [TILDA Textile Dataset](http://lmb.informatik.uni-freiburg.de/resources/datasets/tilda.en.html)

**Pipeline adjustments:**
```python
# 1. Tile-based processing — fabric images are large, defects are small
# Tile 1000x1000 image into 128x128 patches, classify each patch
def tile_image(img, tile_size=128, overlap=16):
    tiles = []
    h, w = img.shape[:2]
    for y in range(0, h-tile_size, tile_size-overlap):
        for x in range(0, w-tile_size, tile_size-overlap):
            tiles.append(img[y:y+tile_size, x:x+tile_size])
    return tiles

# 2. High class imbalance — defects are rare on production lines
# Use oversampling or weighted sampling
# Defect : Good ratio is often 1:20 in real production

# 3. Texture-specific augmentation
layers.RandomRotation(0.5),   # fabric can be rotated — weave pattern is rotation-invariant
layers.RandomFlip("horizontal_and_vertical"),

# 4. Unfreeze more layers — texture needs deep backbone adaptation
for layer in base_model.layers[-60:]:
    layer.trainable = True
```

**GradCAM value:** Pinpoints exact defect location on fabric tile — QC inspector can confirm or override automated rejection. Audit trail for factory records.

**Deployment target:** Industrial camera + Raspberry Pi / Jetson Nano at weaving machine. Real-time TFLite inference.

**Difficulty:** 🟡 Medium — class imbalance is the main challenge

---

### 3.2 PCB Defect Classification

**Problem:** Classify printed circuit board defects in electronics manufacturing.

**Classes:**
- `good`
- `missing_component`
- `solder_bridge`
- `open_circuit`
- `misalignment`

**Dataset:**
- [PCB Defect Dataset — Kaggle](https://www.kaggle.com/datasets/akhatova/pcb-defects) — 1,386 images
- [DeepPCB Dataset — GitHub](https://github.com/tangsanli5201/DeepPCB)

**Pipeline adjustments:**
```python
# 1. Microscopy images — very high resolution, small defects
IMG_SIZE = (224, 224)  # or crop to defect region if bounding boxes available

# 2. No color augmentation — PCB color encodes component type
# REMOVE all color augmentations
# KEEP only spatial: flip, rotation, zoom

# 3. Difference imaging — compare against known-good reference board
def diff_image(test_img, reference_img):
    return cv2.absdiff(test_img, reference_img)
# Feed difference image to classifier for much higher accuracy

# 4. Object detection approach may be better than classification
# Consider YOLOv8 for precise defect localization
# But classification pipeline works for proof-of-concept
```

**GradCAM value:** Shows which board region triggered prediction — electronics engineer can cross-reference with component layout.

**Deployment target:** Electronics factory SMT line camera system.

**Difficulty:** 🟡 Medium

---

### 3.3 Steel Surface Defect Detection

**Problem:** Classify surface defects in steel sheets during rolling mill production.

**Classes:**
- `crazing`
- `inclusion`
- `patches`
- `pitted_surface`
- `rolled_in_scale`
- `scratches`

**Dataset:**
- [NEU Surface Defect Database — Kaggle](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) — 1,800 images, 6 classes, balanced

**Pipeline adjustments:**
```python
# 1. Grayscale images — convert to RGB same as X-ray
# NEU dataset is grayscale

# 2. Texture augmentation — steel surface defects are texture-based
layers.RandomRotation(0.5),
layers.RandomFlip("horizontal_and_vertical"),

# 3. This dataset is perfectly balanced — no class weight adjustment needed
# One of the cleanest industrial datasets available

# 4. Very similar to base pipeline — minimal changes
# Just update class_names and IMG_SIZE
class_names = ['crazing', 'inclusion', 'patches',
               'pitted_surface', 'rolled_in_scale', 'scratches']
```

**GradCAM value:** Shows surface region — steel inspector can compare with physical sample.

**Deployment target:** Rolling mill production line camera.

**Difficulty:** 🟢 Easy — NEU dataset is exceptionally clean and balanced

---

### 3.4 Concrete Crack Detection

**Problem:** Classify structural images for crack detection in civil infrastructure inspection.

**Classes:**
- `no_crack`
- `hairline_crack`
- `structural_crack`
- `spalling`

**Dataset:**
- [Concrete Crack Images — Kaggle](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection) — 40,000 images, balanced
- [SDNET2018](https://digitalcommons.usu.edu/all_datasets/48/)

**Pipeline adjustments:**
```python
# 1. Minimal changes from base pipeline — very clean binary dataset
# Binary version (crack / no crack) achieves ~99% accuracy easily

# 2. For multi-class severity — tile large wall images into patches
# Same tile-based approach as fabric defect

# 3. Color augmentation minimal — concrete is gray
layers.RandomContrast(0.3),   # lighting variation on structures
layers.RandomBrightness(0.2),

# 4. Practically zero pipeline change — closest to the base project
class_names = ['hairline_crack', 'no_crack', 'spalling', 'structural_crack']
```

**GradCAM value:** Highlights crack path — civil engineer can measure extent and plan repair.

**Deployment target:** Drone-mounted camera + edge device for bridge/building inspection. PWD smart infrastructure initiatives.

**Difficulty:** 🟢 Easy — large, balanced, clean dataset

---

## Domain 4 — Environment & Smart Cities

---

### 4.1 Waste Segregation ⭐ Top Pick

**Problem:** Smart dustbin or sorting belt classifier for municipal waste management.

**Classes:**
- `plastic`
- `paper`
- `metal`
- `organic`
- `glass`
- `e_waste`

**Dataset:**
- [Garbage Classification — Kaggle](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification) — 2,527 images
- [TrashNet](https://github.com/garythung/trashnet) — 2,527 images, 6 classes

**Pipeline adjustments:**
```python
# 1. Virtually zero pipeline change from base project
# Just rename classes and retrain

# 2. Add slight color augmentation — lighting in bins varies
layers.RandomBrightness(0.3),
layers.RandomContrast(0.3),

# 3. Background variation augmentation — items appear on different surfaces
# Consider CutMix augmentation to handle partial occlusion

# 4. Deploy as IoT device — TFLite on Raspberry Pi + servo motor for bin lid
# This is the complete deployment story for this project
class_names = ['e_waste', 'glass', 'metal', 'organic', 'paper', 'plastic']
```

**GradCAM value:** Shows which material property (texture, surface finish) was used — helps debug misclassifications like crumpled metallic paper.

**Deployment target:** Smart dustbin (Raspberry Pi + camera + servo). Swachh Bharat alignment.

**Difficulty:** 🟢 Easy

---

### 4.2 Road Damage Classification

**Problem:** Automated road condition monitoring for smart city maintenance prioritisation.

**Classes:**
- `good`
- `pothole`
- `longitudinal_crack`
- `transverse_crack`
- `waterlogging`
- `road_marking_faded`

**Dataset:**
- [Road Damage Dataset — Kaggle](https://www.kaggle.com/datasets/sovitrath/road-damage-dataset)
- [RDD2022 — IEEE DataPort](https://ieee-dataport.org/competitions/road-damage-detection-using-smartphone-images)

**Pipeline adjustments:**
```python
# 1. Minimal changes from base pipeline
# Road images are RGB, similar scale to base dataset

# 2. Weather augmentation — roads are photographed in rain, sun, night
layers.RandomBrightness(0.4),   # day vs night
layers.RandomContrast(0.3),

# 3. Motion blur augmentation — images taken from moving vehicles
import tensorflow_addons as tfa
def random_motion_blur(img):
    return tfa.image.translate(img, [np.random.randint(-5,5), 0])

# 4. GPS tagging in Gradio app — add lat/lon input field
# Map integration shows damage locations on Punjab road network
```

**GradCAM value:** Highlights damage region — PWD field engineer gets precise location to inspect and estimate repair cost.

**Deployment target:** Citizen reporting app, municipality vehicle dashcam system.

**Difficulty:** 🟢 Easy

---

### 4.3 Air Quality / Sky Haze Classification

**Problem:** Classify sky condition from CCTV or phone camera for AQI monitoring.

**Classes:**
- `clear`
- `hazy`
- `foggy`
- `smoggy`
- `dust_storm`

**Dataset:**
- [RESIDE Haze Dataset — Kaggle](https://www.kaggle.com/datasets/balraj98/indoor-training-set-its-residestandard)
- Build partial dataset from CPCB CCTV feeds (Punjab)

**Pipeline adjustments:**
```python
# 1. Sky region crop — bottom half of image (ground) is irrelevant
# Crop top 60% of image before feeding to model
def sky_crop(img):
    h = img.shape[0]
    return img[:int(h*0.6), :, :]

# 2. Time-of-day normalisation — same location looks different at dawn vs noon
layers.RandomBrightness(0.5),   # very wide range
layers.RandomContrast(0.4),

# 3. Temporal smoothing for continuous monitoring
# Average predictions over 5-minute windows to reduce noise

# 4. Regression option — continuous haze score (0.0 clear → 1.0 opaque)
# More useful than discrete classes for AQI correlation
```

**GradCAM value:** Confirms model is reading sky region and not foreground buildings — important for calibrating against AQI sensor data.

**Deployment target:** CPCB monitoring stations, municipal CCTV network. Punjab stubble burning season monitoring.

**Difficulty:** 🟡 Medium — dataset assembly is the hard part

---

### 4.4 Flood Damage Severity

**Problem:** Classify aerial/satellite images for disaster response and resource prioritisation.

**Classes:**
- `no_damage`
- `minor_flooding`
- `major_flooding`
- `building_destroyed`
- `road_cut_off`

**Dataset:**
- [FloodNet — Kaggle](https://www.kaggle.com/datasets/naivelamb/floodnet) — post-Harvey aerial images
- [xBD Disaster Dataset](https://xview2.org/dataset)

**Pipeline adjustments:**
```python
# 1. Input size increase — aerial images need higher resolution context
IMG_SIZE = (224, 224)

# 2. No horizontal flip restriction — aerial views have no canonical orientation
layers.RandomFlip("horizontal_and_vertical"),
layers.RandomRotation(0.5),

# 3. Multi-scale processing — flood extent appears at different zoom levels
# Consider image pyramid approach

# 4. Pre-trained satellite weights if available
# Alternatively, use ResNet50 (better for aerial imagery than MobileNetV2)
base_model = tf.keras.applications.ResNet50(
    include_top=False, weights='imagenet', input_shape=(224,224,3)
)

# 5. Unfreeze more layers — aerial imagery domain gap from ImageNet is larger
for layer in base_model.layers[-80:]:
    layer.trainable = True
```

**GradCAM value:** Shows flooded vs dry regions — NDRF coordinator can verify before deploying rescue resources.

**Deployment target:** State disaster management authority dashboard. Drone feed analysis during monsoon.

**Difficulty:** 🟡 Medium

---

## Domain 5 — Social Impact & Accessibility

---

### 5.1 Handwritten Gurmukhi Character Recognition ⭐ Top Pick

**Problem:** OCR foundation for digitising handwritten government records, school documents in Punjab.

**Classes:** 35 Gurmukhi base characters (ਸ, ਹ, ਕ, ਖ, ...)

**Dataset:**
- [Handwritten Gurmukhi Characters — Kaggle](https://www.kaggle.com/datasets/iamsouravbanerjee/handwritten-gurmukhi-character-dataset)
- [HP Labs Gurmukhi Dataset](http://www.hpl.hp.com/india/research/penhw-resources.html)

**Pipeline adjustments:**
```python
# 1. Binary/grayscale input — handwriting is black on white
# Convert to grayscale, then replicate to 3 channels
def preprocess_handwriting(img):
    gray = tf.image.rgb_to_grayscale(img)
    return tf.image.grayscale_to_rgb(gray)

# 2. Morphological augmentation — simulates different pen pressures
import cv2
def random_dilate_erode(img):
    kernel = np.ones((2,2), np.uint8)
    if np.random.random() > 0.5:
        return cv2.dilate(img, kernel, iterations=1)   # thicker strokes
    return cv2.erode(img, kernel, iterations=1)        # thinner strokes

# 3. Elastic distortion — simulates natural handwriting variation
# Use imgaug library: iaa.ElasticTransformation(alpha=50, sigma=5)

# 4. No color augmentation — handwriting has no color information

# 5. 35-class output — update final dense layer
Dense(35, activation='softmax')  # one per Gurmukhi character

# 6. Confusion matrix is critical — some characters look visually similar
# e.g., ਹ vs ਣ need special attention in evaluation
```

**GradCAM value:** Shows which stroke segment was decisive — useful for debugging similar-looking character pairs and improving training data for confusing classes.

**Deployment target:** Government document digitisation pipeline. School admission form scanner. NIELIT certificate automation.

**Difficulty:** 🟡 Medium — unique regional contribution, moderate dataset size

---

### 5.2 Indian Currency Note Classification

**Problem:** Assistive denomination reader for visually impaired users.

**Classes:**
- `rs_10`
- `rs_20`
- `rs_50`
- `rs_100`
- `rs_200`
- `rs_500`

**Dataset:**
- [Indian Currency Notes — Kaggle](https://www.kaggle.com/datasets/vishalmane109/indian-currency-note-images-dataset-2020)
- Build own dataset: 50–100 photos per denomination with phone camera

**Pipeline adjustments:**
```python
# 1. Virtually zero change from base pipeline — cleanest possible problem
# Notes are photographed under controlled conditions

# 2. Add perspective distortion augmentation — notes are held at angles
import cv2
def random_perspective(img):
    h, w = img.shape[:2]
    pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    offset = 20
    pts2 = pts1 + np.random.uniform(-offset, offset, pts1.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (w,h))

# 3. Crumpled/worn note augmentation — real currency is not pristine
layers.RandomContrast(0.4),
layers.RandomBrightness(0.4),

# 4. TFLite deployment is the key deliverable here
# Visually impaired users need offline, fast inference
# Target < 100ms per prediction on mid-range phone
```

**GradCAM value:** Shows which denomination feature (Gandhi portrait position, numeral region, colour band) was used — useful for debugging misclassifications of worn notes.

**Deployment target:** Android accessibility app. Screen reader integration.

**Difficulty:** 🟢 Easy — one of the easiest problems in this list

---

### 5.3 Sign Language (Indian ISL) Recognition

**Problem:** Classify Indian Sign Language hand gestures for communication assistance.

**Classes:** 26 ISL alphabet signs + 10 digit signs

**Dataset:**
- [Indian Sign Language — Kaggle](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl)
- [ISL Dataset — Kaggle](https://www.kaggle.com/datasets/vaishnaviasonawane/indian-sign-language-dataset)

**Pipeline adjustments:**
```python
# 1. Hand segmentation preprocessing — remove background
# Use MediaPipe Hands to crop to hand bounding box before classification
import mediapipe as mp
mp_hands = mp.solutions.hands

# 2. Skin tone robustness — model must work across skin tones
# Ensure diverse training data
# Consider YCbCr color space for skin segmentation

# 3. Mirror augmentation — ISL signs can be performed with either hand
layers.RandomFlip("horizontal"),

# 4. Lighting robustness — used in varied environments
layers.RandomBrightness(0.4),
layers.RandomContrast(0.3),

# 5. Real-time webcam inference in Gradio
# Add video input instead of image upload
interface = gr.Interface(
    fn=predict,
    inputs=gr.Video(source="webcam"),
    outputs=[gr.Label(num_top_classes=3), gr.Image()]
)
```

**GradCAM value:** Highlights hand region and finger configuration — helps verify model isn't using background or sleeve color.

**Deployment target:** Real-time desktop/mobile app for deaf community communication.

**Difficulty:** 🟡 Medium

---

### 5.4 Document Type Classification

**Problem:** Automatically route scanned government documents in digitisation workflows.

**Classes:**
- `aadhaar_card`
- `pan_card`
- `ration_card`
- `birth_certificate`
- `land_record`
- `school_certificate`

**Dataset:**
- Build custom dataset — scan or photograph each document type
- ~200 images per class is sufficient for this problem (very distinct layouts)
- RVL-CDIP dataset for general document types

**Pipeline adjustments:**
```python
# 1. Grayscale to RGB conversion — scanned documents are often grayscale

# 2. Minimal spatial augmentation — document orientation matters
layers.RandomFlip("horizontal"),  # horizontal only — upside-down documents rare
layers.RandomRotation(0.05),      # very small rotation — documents are mostly aligned
layers.RandomZoom(0.1),

# 3. Deskewing preprocessing — scanned documents may be slightly rotated
import cv2
def deskew(img):
    coords = np.column_stack(np.where(img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45: angle = -(90 + angle)
    else: angle = -angle
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

# 4. Layout-aware features are more important than texture
# MobileNetV2 captures layout patterns well — no backbone change needed
```

**GradCAM value:** Shows which region (header logo, specific form field, photograph area) was used — confirms model reads document structure not paper texture.

**Deployment target:** Government e-Governance document processing pipeline. NIELIT certificate verification system.

**Difficulty:** 🟢 Easy — document types are visually very distinct

---

## Domain 6 — Wildlife & Nature

---

### 6.1 Wildlife Camera Trap Classification

**Problem:** Auto-sort thousands of camera trap images from forest departments — replacing manual review.

**Classes:**
- `deer`
- `leopard`
- `wild_boar`
- `elephant`
- `empty_frame`
- `human`

**Dataset:**
- [iWildCam — Kaggle](https://www.kaggle.com/competitions/iwildcam-2022-fgvc9)
- [Snapshot Serengeti](https://www.kaggle.com/datasets/lila-science/snapshot-serengeti)
- [Wildlife Protection Dataset — Kaggle](https://www.kaggle.com/datasets/brsdincer/wildlife-protection-dataset)

**Pipeline adjustments:**
```python
# 1. Night vision / IR images — many camera trap photos are infrared
# Convert IR (near-black-and-white) to 3-channel input
# Treat same as grayscale → RGB conversion

# 2. Extreme class imbalance — empty frames dominate (60-70% of images)
# Use weighted sampling or oversample rare animal classes
class_weight = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.3, 5: 5.0}
# empty_frame weight=0.3, human weight=5.0 (safety critical)

# 3. Background variation is extreme — same species in jungle, grassland, stream
# Use more aggressive augmentation
layers.RandomBrightness(0.5),
layers.RandomContrast(0.5),
layers.RandomSaturation(0.3),

# 4. Human class is safety-critical — high recall required
# Tune threshold: flag human if P(human) > 0.3 (not 0.5)
```

**GradCAM value:** Shows animal body vs background branches/leaves — critical for reducing false positives from vegetation movement.

**Deployment target:** Forest department camera network. Automated alert system for human-wildlife conflict zones.

**Difficulty:** 🟡 Medium — class imbalance and IR images are the main challenges

---

### 6.2 Bird Species Classification

**Problem:** Identify bird species from photographs for biodiversity monitoring.

**Classes:** Start with 10–20 species common to Punjab/Himachal region

**Dataset:**
- [CUB-200-2011 Birds — Kaggle](https://www.kaggle.com/datasets/veeralakrishna/200-bird-species-with-11788-images)
- [Birds 525 Species — Kaggle](https://www.kaggle.com/datasets/gpiosenka/100-bird-species) — 525 species, 90k images

**Pipeline adjustments:**
```python
# 1. Fine-grained classification — species differences are subtle
# Need higher resolution
IMG_SIZE = (224, 224)

# 2. Unfreeze more backbone layers — fine-grained needs deeper adaptation
for layer in base_model.layers[-80:]:
    layer.trainable = True

# 3. More Phase 2 epochs — fine-grained needs longer fine-tuning
PHASE2_EPOCHS = 30

# 4. Consider EfficientNetB4 backbone for fine-grained tasks
# MobileNetV2 may underfit on fine-grained species differences
base_model = tf.keras.applications.EfficientNetB4(
    include_top=False, weights='imagenet', input_shape=(224,224,3)
)

# 5. Part-based attention (advanced) — focus on beak, wing pattern, tail
# Grad-CAM helps identify which parts the model focuses on
```

**GradCAM value:** Shows which bird body part (beak shape, wing pattern, tail length) was used — ornithologists can verify taxonomically.

**Deployment target:** Citizen science app (like eBird India). Forest department biodiversity survey tool.

**Difficulty:** 🔴 Hard — fine-grained classification, many visually similar species

---

## Quick Reference — Pipeline Change Summary

| Project | IMG Size | Flip | Color Aug | Backbone | Output Neurons | Key Extra Step |
|---|---|---|---|---|---|---|
| Crop Disease | 224×224 | Both | + Hue, Sat | MobileNetV2 | 4–5 | CLAHE optional |
| Fruit Quality | 128×128 | Both | + Hue, Sat | MobileNetV2 | 3 | Background removal |
| Soil Type | 128×128 | Both | Aggressive | MobileNetV2 | 5 | More epochs |
| Rice Grain | 224×224 | Both | None extra | MobileNetV2 | 5 | Center crop |
| Eye Disease | 224×224 | H only | None | EfficientNetB3 | 5 | CLAHE, class weights |
| Skin Triage | 224×224 | H only | + Hue, Bright | MobileNetV2 | 5 | Focal loss |
| Yoga Pose | 224×224 | H only | None | MobileNetV2 | 7 | Bg removal optional |
| X-Ray | 224×224 | H only | None | DenseNet121 | 4 | Grayscale→RGB |
| Fabric Defect | 128×128 | Both | None | MobileNetV2 | 6 | Tiling |
| PCB Defect | 224×224 | Both | None | MobileNetV2 | 5 | Diff imaging |
| Steel Defect | 128×128 | Both | None | MobileNetV2 | 6 | Grayscale→RGB |
| Concrete Crack | 128×128 | H only | + Bright | MobileNetV2 | 4 | Tiling for large images |
| Waste Sort | 128×128 | Both | + Bright | MobileNetV2 | 6 | None |
| Road Damage | 128×128 | H only | + Bright | MobileNetV2 | 6 | None |
| Air Quality | 128×128 | H only | + Bright | MobileNetV2 | 5 | Sky crop |
| Flood Damage | 224×224 | Both | None | ResNet50 | 5 | Unfreeze 80 layers |
| Gurmukhi OCR | 128×128 | H only | None | MobileNetV2 | 35 | Grayscale, elastic distort |
| Currency | 128×128 | Both | + Bright | MobileNetV2 | 6 | Perspective distort |
| Sign Language | 224×224 | H only | + Bright | MobileNetV2 | 36 | Hand segmentation |
| Document Type | 128×128 | H only | None | MobileNetV2 | 6 | Deskewing |
| Camera Trap | 128×128 | Both | + Bright | MobileNetV2 | 6 | Class weights, IR handling |
| Bird Species | 224×224 | H only | + Sat | EfficientNetB4 | 20+ | Unfreeze 80 layers |

---

## Choosing Your Next Project — Decision Guide

```
Is there a real user who is not a tech person?
├── Yes → Agriculture / Health / Social Impact domain (Tier 1 impact)
└── No  → Industry / Environment (still good, but academic framing)

Is the dataset free and >5,000 images?
├── Yes → Go for it immediately
└── No  → Budget 2–3 weeks for dataset assembly / augmentation

Does GradCAM add trust value beyond demo?
├── Yes → Medical / Industrial (the explainability IS the feature)
└── No  → Fine for portfolio, but weaker real-world deployment story

Is edge deployment (TFLite) part of the story?
├── Yes → Anything in Agriculture, Waste, Currency, Sign Language
└── No  → Web-only Gradio app is sufficient
```

---

## Author

**Lovnish Verma** — Project Engineer, NIELIT Ropar  
AIML Programme in collaboration with IIT Ropar

[github.com/lovnishverma](https://github.com/lovnishverma) · [huggingface.co/lovnishverma](https://huggingface.co/lovnishverma) · [kaggle.com/princelv84](https://www.kaggle.com/princelv84)
