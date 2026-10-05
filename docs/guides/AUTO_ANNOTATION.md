# Auto-annotation (model-assisted labelling) - how it works here and how to use it

**Idea.** Don't label every image by hand. Train on a small seed, let the model *propose* labels for the rest, then
a person reviews the proposals, most doubtful first. The reviewed labels retrain a better model, and the loop
repeats. Published industrial-inspection work reports that accepting, rejecting or correcting proposals can save
more than half of the annotation time, and that "inspection by exception" (asking a person only about the images the
model is least sure of) reaches good results with a fraction of the labels.

**The one rule kept from the rest of this system:** a model's proposal is never training data. It sits beside the
image (`cache/suggestions.json` for defect labels, `"proposals"` in the annotation file for boxes) until a person
accepts it. An image stays *not reviewed* and out of training until then.

## 0. The loop that makes the model better (built 2026-10-06)

```
Live page (engineer): Auto-collect ON
   -> each DECIDED bottle (one steady verdict, verdict.py) -> ONE frame into the Label inbox, NOT reviewed,
      with the model's defect scores as a suggestion
Label page: filter "AI suggested - not reviewed" -> accept (Enter) / correct (1-9, G) / bulk-accept confident ones
   -> every disagreement with the model is stored as a HARD example (cache/hard_examples.json):
      false_defect (model said defect, person said GOOD), missed_defect, wrong_defect
Train: good bottles drawn more often (up to x5) + hard examples x3 more  -> a CANDIDATE
Models: Validate = real-camera numbers within the limits + background-shortcut check -> Approve -> ACTIVATE
```

- Auto-collect needs the detector (Live mode *Classifier + YOLO*): the detector is what says a bottle arrived and
  left, so exactly one frame is saved per bottle. A daily cap (`autocollect_daily_cap`, default 300) stops runaway
  collection. Nothing collected is ever training data until a person reviews it.
- Unsure bottles: a defect score just under its threshold shows **CHECK: <defect>?** on screen (`check_margin`,
  default 0.10) instead of GOOD or a named defect. On the line the same band is available as
  `decision_rules.check_margin` (off by default; when on, such a bottle is FAULT and rejected).
- Activation limits (`settings.json` `activation_gates`): good bottles called defective <= 2 %, defective bottles
  passed <= 1 %, at least 30 real good and 30 real defective bottles shown, and capped white-background bottles
  called missing_cap <= 10 % (the shortcut check).

## 1. Defect labels (the classifier names the defect)

For deciding GOOD / DEFECT and *which* defect (missing cap, tilted cap, damaged label ...):

1. **Label** page -> put new images in the inbox (drop them in `-ve/` without a class folder, or Import).
2. Press **AI pre-label**. The active classifier scores every inbox image. Each image shows the defect *names* it
   predicts, e.g. "Missing cap 0.92" (Label inspector, "AI" line).
3. Filter **"AI suggested - not reviewed"**. **Enter** accepts the suggestion, **1-9** corrects it, **G** = good.
4. **Accept all confident AI labels in this view** accepts only images where every score is >= 95 % or <= 5 % and at
   most one defect is flagged. Spot-check a few first.
5. Train again (Train page). The new model becomes a CANDIDATE; activate it on the Models page after validation.

A defect fires when its score reaches the *calibrated* threshold (`python calibrate_thresholds.py`), not a hand-tuned
low one, so a good bottle is not proposed as defective by noise.

## 2. Component boxes (the detector draws bottle / cap / label)

For training the YOLO detector (it finds parts; the decision engine turns a *missing* part into a defect):

1. **Annotate** page -> **Propose boxes (model)** runs the detector on the unlabelled images.
2. Boxes appear as dashed proposals, and the page shows **PREDICTED: GOOD / DEFECT: Missing cap / NO BOTTLE FOUND**
   (the same rule the line uses), so you see what the boxes *mean*, not just where they are. **Accept proposals** / **Reject proposals** per image; fix a box by hand.
3. Work the **active-learning queues** instead of going in file order:
   - **Next: least sure** - lowest confidence (or nothing found).
   - **Next: missing part** - a bottle was found but no cap or label: the likely *real* missing-cap / missing-label
     bottles, rare and valuable.
   - **Next: doubtful part** - a cap or label proposed at 0.25-0.80 confidence: hard negatives, e.g. a bare neck's
     green tamper ring read as a cap.
4. Mark the image *reviewed*, then export (YOLO detection) and train a candidate (`model_bench.py yolo --data v3`).

## 3. What auto-annotation cannot do

- It cannot fix a *bias*: if every missing-cap photo is on a white background, a model learns "white background",
  and its own proposals repeat the mistake (this happened on 2026-10-05, see VISION_DATASET.md). Always include
  missing-cap bottles from several bottles and poses in the real setup.
- It cannot judge a defect it has never seen. For a rare defect, capture real examples first (Live -> Capture frame).
- Proposals are guesses; the person is the ground truth.

## Sources

- Labelbox, [How to build defect detection models to automate visual quality inspection](https://labelbox.com/guides/how-to-build-defect-detection-models-to-improve-visual-quality-inspection/)
- [Active Learning for Automated Visual Inspection of Manufactured Products](https://arxiv.org/html/2109.02469v1)
- [Active learning for industrial defect detection: a study on hybrid sampling strategies](https://link.springer.com/article/10.1007/s00170-025-17378-7)
- [Model-Assisted Labeling via Explainability for Visual Inspection of Civil Infrastructures](https://arxiv.org/pdf/2209.11159)
