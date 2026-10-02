"""Generates the Stage 1 benchmark charts for the FINAL_YEAR_BLACKBOOK from
the actual values recorded in projects/om_bottle/models/<stamp>/metrics.json
and the confirmed held-out test evaluations. No values are invented here --
every number below is copied from a metrics.json field or from a test-set
evaluation already reported and confirmed in the Stage 1 close-out.

Read-only with respect to the ML project: this script only reads
metrics.json files and writes PNGs into this documentation folder.
"""
import matplotlib.pyplot as plt

MODELS = ["EfficientNet-B0", "EfficientNet-B1", "MobileNetV3-Small", "ResNet18"]
COLORS = ["#2E7D32", "#1565C0", "#B08900", "#8E24AA"]

test_macro_f1 = [0.935, 0.845, 0.632, 0.773]
exact_match = [89.1, 68.1, 67.2, 84.0]          # percent
total_fp = [28, 74, 82, 38]
total_fn = [0, 10, 36, 12]
test_fps = [72.0, 56.5, 71.2, 59.8]
model_size_mb = [15.57, 25.24, 5.93, 42.72]
training_time_s = [216.2, 453.9, 192.2, 145.3]


def bar_chart(values, ylabel, title, fname, fmt="{:.3f}", ymax=None):
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=150)
    bars = ax.bar(MODELS, values, color=COLORS, edgecolor="black", linewidth=0.6)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    if ymax:
        ax.set_ylim(0, ymax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=15, ha="right")
    for b, v in zip(bars, values):
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


bar_chart(test_macro_f1, "Test macro-F1", "Held-out test macro-F1 (238 images, val-derived thresholds)",
          "model_test_macro_f1.png", fmt="{:.3f}", ymax=1.05)
bar_chart(exact_match, "Exact-match accuracy (%)", "Held-out test exact-match accuracy (238 images)",
          "model_exact_match_accuracy.png", fmt="{:.1f}%", ymax=100)
bar_chart(total_fp, "Total false-positive label instances", "Held-out test false positives (238 images)",
          "model_false_positive_comparison.png", fmt="{:.0f}")
bar_chart(total_fn, "Total false-negative label instances", "Held-out test false negatives (238 images)",
          "model_false_negative_comparison.png", fmt="{:.0f}")
bar_chart(test_fps, "Inference FPS (live-scoring methodology)", "Held-out test inference speed",
          "model_inference_fps.png", fmt="{:.1f}")
bar_chart(model_size_mb, "Model size (MB)", "Checkpoint (model.pt) size",
          "model_size_comparison.png", fmt="{:.2f}")
bar_chart(training_time_s, "Training time (s)", "Total training time (fresh process, CUDA)",
          "training_time_comparison.png", fmt="{:.1f}")

print("charts written")
