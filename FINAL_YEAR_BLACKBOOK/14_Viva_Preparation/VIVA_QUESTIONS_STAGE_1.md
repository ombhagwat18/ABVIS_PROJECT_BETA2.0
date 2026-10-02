# Viva Questions — Stage 1

## 1. Why did you use scene-based splitting instead of a random split?

The dataset is made of consecutive video frames of a small number of
physical bottle passes, so frames within one pass look nearly identical. A
random per-image split would put near-duplicate frames of the same bottle
pass into both training and test, letting the model "pass" the test by
recognising a frame it effectively already memorised in training. Grouping
frames into scenes and splitting whole scenes keeps every frame of a given
bottle pass in only one partition, so the test score reflects
generalisation to a genuinely unseen bottle, not memorisation.

## 2. Why do you need train, validation, AND test sets — why not just train/test?

Training data teaches the model. Validation data is used during training
to pick the best epoch and the per-class decision thresholds, without
touching the data used for the final honest measurement. If thresholds and
model selection were done on the test set, the test score would no longer
be an honest estimate of generalisation — it would be a number optimised to
look good on that particular set of images. The held-out test set is
touched exactly once, after every training decision has already been made.

## 3. Why is this a multi-label classification problem, not multi-class?

The defect classes are not mutually exclusive — a bottle could, in
principle, have more than one defect at once (e.g. skewed and
label-damaged). Multi-class classification forces exactly one label per
image, which does not match that reality, even though the current dataset
happens to contain no image with more than one label set.

## 4. Why EfficientNet as a starting architecture?

EfficientNet-B0 is a widely used, ImageNet-pretrained convolutional
backbone that balances accuracy and computational cost, making it a
reasonable practical default for a project without a large in-domain
dataset to train a backbone from scratch on. It was the first architecture
trained and evaluated, and was then compared against three others to check
whether a different architecture would generalise better.

## 5. Why compare multiple architectures instead of using just one?

Accepting a single architecture's result without comparison risks
mistaking one model's quirks for the true ceiling of what is achievable, or
missing a materially better option. Comparing EfficientNet-B0,
EfficientNet-B1, MobileNetV3-Small, and ResNet18 under an identical
protocol produces an evidence-based baseline choice rather than an
assumption.

## 6. Why was EfficientNet-B0 selected as the baseline?

Under the evaluated protocol, it achieved the highest held-out test
macro-F1 (0.935), the highest exact-match accuracy (89.1%), and was the
only one of the four architectures with zero observed false-negative label
instances on the 238-image held-out test set — the most consequential
error type for a pass/fail QC decision.

## 7. What is macro-F1, and why report it instead of plain accuracy?

Macro-F1 is the unweighted average of the F1 score (the harmonic mean of
precision and recall) computed independently for each class, then
averaged. Plain accuracy is misleading under class imbalance: a model that
always predicts "no defect" can score very high accuracy on a rare defect
class while never detecting it. Macro-F1 gives every class equal weight
regardless of how common it is, which better reflects whether rare defect
classes are actually being detected.

## 8. Define precision and recall in this context.

Precision = TP / (TP + FP): of all the times the model raised a given
defect flag, what fraction were actually correct. Recall = TP / (TP + FN):
of all the bottles that actually had that defect, what fraction the model
caught. High precision means few false alarms; high recall means few
missed defects.

## 9. What is a false positive vs. a false negative here?

A false positive is a genuinely good bottle (or a bottle without that
specific defect) that the model incorrectly flags with a defect — causing
an unnecessary rejection. A false negative is a bottle that genuinely has a
defect that the model misses and passes as good — the defect ships to the
customer.

## 10. Why is a false negative considered more serious than a false positive in QC?

A false positive costs a wasted rejection (a throughput/cost problem). A
false negative lets a defective product reach the customer (a quality and,
potentially, safety/liability problem). This is why EfficientNet-B0's zero
observed false negatives on the held-out test set is highlighted separately
from its overall F1 score.

## 11. Why must the held-out test set remain completely untouched during training?

If the test set influences any training decision — which epoch to keep,
which architecture to prefer, or which threshold to use — its score stops
measuring generalisation and starts measuring how well the model was
tuned to that specific set of images. The whole point of a held-out test
set is to answer "how would this model perform on bottles it has never
influenced its own configuration with."

## 12. Why are the decision thresholds derived from validation, not from the test set?

The threshold decides, per class, at what predicted probability a defect is
declared present. Choosing this threshold using test labels would let the
model's reported test score be inflated by a threshold hand-picked to fit
that specific test set — an unfair, non-reproducible advantage. Choosing it
from the validation set instead keeps the test evaluation honest.

## 13. Why is `missing_cap` unscored rather than given a low score?

`missing_cap` has zero positive images anywhere in the dataset. A "0.000"
score would misleadingly suggest the model was tested and failed; in fact
it was never tested at all, because no example exists. Reporting it as
explicitly unscored, and disabling its detection threshold, avoids
implying an untrue claim in either direction.

## 14. Why did `missing_label` fail for three of the four models?

`missing_label` has only 30 images across 2 scenes in the whole dataset,
and the deterministic split places 0 of those scenes into validation. Every
model's threshold for this class was therefore chosen with no validation
signal to check it against; for three of the four architectures that
untested threshold proved unreachable on real test-time probabilities,
producing 0/12 test recall.

## 15. Why can validation performance differ so much from test performance?

Validation is a small (220-image, 19-scene) sample used repeatedly during
training to pick the best epoch, which risks some degree of implicit
overfitting to that specific sample, especially with few scenes. The
held-out test set, touched only once, can reveal that a model generalises
worse than its validation score suggested — exactly what was observed for
EfficientNet-B1 (validation macro-F1 1.000, test macro-F1 0.845).

## 16. Why doesn't a 93.5% test macro-F1 mean 93.5% real-world accuracy?

The 93.5% figure describes performance on one specific 238-image, 23-scene
held-out test set, under one specific camera, lighting, ROI, and bottle
population. Real production conditions — different lighting, camera drift,
different bottle batches — are not represented in this dataset, so the
figure is a measured result on this test set, not a guaranteed production
accuracy.

## 17. Why use a GPU for training?

Training a convolutional neural network involves millions of matrix
multiplications per batch; a GPU parallelises these far more efficiently
than a CPU, reducing training time from potentially hours to minutes per
run — relevant here since four full architecture experiments were run.

## 18. Why measure inference FPS, not just accuracy?

The system is intended for a live QC line, where inspection has to keep
pace with bottles passing a camera. A highly accurate model that is too
slow to run in real time would not be usable in the intended deployment,
so throughput (FPS) and latency are reported alongside accuracy metrics.

## 19. Why does model size matter?

Model size affects memory footprint, load time, and deployability
(e.g. on constrained hardware). It was recorded for every architecture
(5.93 MB to 42.72 MB across the four models evaluated) as one factor in the
overall comparison, though it was not the deciding factor for the Stage 1
baseline selection.

## 20. What are the main dataset limitations identified in Stage 1?

Zero positive images for `missing_cap`; a thin `missing_label` class (30
images, 2 scenes, 0 validation positives); no genuine multi-defect images
in the dataset; and a modest total scene count (112 scenes behind 1,143
images), which bounds the statistical confidence of every reported metric.

## 21. What is planned for Stage 2?

Planning a universal annotation and object-detection (YOLO) capability.
No Stage 2 implementation has begun; Stage 1 concluded with architecture
selection and documentation only.

## 22. What is the difference between classification and object detection?

Classification answers "what is present in this image" for the image as a
whole (or per pre-defined class), without saying where. Object detection
additionally localises each detected object with a bounding box, and can
detect multiple instances of objects within a single image.

## 23. What is the difference between object detection and segmentation?

Object detection outputs a bounding box per detected object. Segmentation
outputs a pixel-level mask, classifying every pixel in the image (or every
pixel within a detected region), giving the exact shape/extent of an
object or defect rather than just its approximate box.

## 24. Why not use YOLO (object detection) for everything, instead of a classifier?

Object detection requires bounding-box annotated data, which does not yet
exist for this dataset, and is only clearly justified when a frame could
contain more than one bottle or when the exact spatial location of a defect
matters for downstream logic (e.g. robotic picking). For a single bottle
per frame with a defect-present/absent decision, whole-image classification
is simpler to annotate, train, and evaluate, and was judged sufficient for
Stage 1's scope. This is a scope decision, not a claim that detection is
unnecessary for later stages — see `15_Future_Work/FUTURE_WORK.md`.

## 25. How would the AI system eventually connect to a PLC?

This has not been implemented. The intended direction (documented as future
work, not completed work) is for the AI inspection result (PASS/FAIL, and
potentially which defect) to be communicated to a PLC-controlled actuation
stage — for example, triggering a pneumatic reject mechanism when a bottle
on the conveyor is classified as defective, likely via a standard
industrial communication protocol. No such integration exists in the
current codebase, and its concrete design (protocol choice, timing/latency
budget, fail-safe behaviour) remains to be defined.

## 26. Why was early stopping added, and only for ResNet18?

Early stopping was added as an opt-in feature (`--patience`) to stop
training once validation macro-F1 stops improving, saving training time
without changing which checkpoint gets selected (still the best validation
epoch). It was introduced after observing, from the first three
experiments, that models tended to reach their best validation epoch well
before epoch 25 (e.g. epoch 3 for EfficientNet-B1, epoch 7 for
MobileNetV3-Small) — so it was applied to the ResNet18 experiment to test
and demonstrate the capability, without retroactively changing the
already-completed B0/B1/MobileNetV3-Small experiments.

## 27. How do you know the four experiments used exactly the same data split?

The scene-based split is deterministic (fixed seed = 0), computed from the
dataset's labels and folder layout at the time each experiment ran. Because
the dataset was not modified between experiments, the `val_paths` and
`test_paths` path lists recorded in each checkpoint's `metrics.json` were
directly compared and confirmed to be byte-identical across all four
checkpoints.
