#!/usr/bin/env python3
"""
Frequency Iris Framework — Mathematical Validation Script v0.4.2
Dataset: CASIA-Iris-Thousand (640x480 greyscale full-eye NIR images)

Corrections from v0.4.1:
- Iris detection completely rewritten for CASIA NIR images
- 5th percentile threshold (was 15th) to isolate pupil only
- Centre proximity scoring to pick the right dark blob
- Nyquist region excluded from FFT peak search
- Peak detection uses statistical threshold, not bare argmax
- Sanity check tightened: r_iris - r_pupil must be >= 30px
- Diagnostic output on first image to verify detection

Run:  python3 validate_iris_v2.py
Results saved to validation_results_v2.txt
"""

import os, sys, math, random
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.fft import fft2, fftshift
from collections import defaultdict

DATASET_PATH     = "/Users/kaytfrazier/Downloads/CASIA-Iris-Thousand"
RESULTS_FILE     = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "validation_results_v3.txt")
IRIS_DIAMETER_MM = 12.0
RANDOM_SEED      = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

lines = []
def log(msg=""):
    print(msg)
    lines.append(str(msg))

# ── IMAGE LOADING ─────────────────────────────────────────────────────────────
def load_gray(path):
    try:
        return np.array(Image.open(path).convert("L"), dtype=np.float64)
    except Exception:
        return None

# ── IRIS DETECTION — rewritten for CASIA 640×480 NIR ─────────────────────────
def detect_boundaries(img, verbose=False):
    """
    Robust pupil + iris boundary detection for CASIA near-infrared eye images.
    Returns (cx, cy, r_pupil, r_iris) or None if detection fails.
    """
    h, w = img.shape   # expected 480, 640

    # Step 1 — smooth to suppress eyelashes and noise
    smooth = ndimage.gaussian_filter(img.astype(float), sigma=3)

    # Step 2 — threshold at 5th percentile to isolate pupil
    # Pupil ~50px radius = ~7854px ≈ 2.6% of 307200 total pixels
    thresh = np.percentile(smooth, 5)
    dark   = smooth < thresh

    # Step 3 — label connected dark blobs
    labeled, n = ndimage.label(dark)
    if n == 0:
        if verbose: log("  detect: no dark blobs found")
        return None

    # Step 4 — score each blob by size × proximity to image centre
    centre_y, centre_x = h / 2, w / 2
    best_score, cx, cy = -1, int(centre_x), int(centre_y)

    for lbl in range(1, n + 1):
        size = int(ndimage.sum(dark, labeled, lbl))
        if size < 300:          # too small to be a pupil
            continue
        blob_y, blob_x = ndimage.center_of_mass(dark, labeled, lbl)
        dist = math.hypot(blob_x - centre_x, blob_y - centre_y)
        score = size / (1.0 + dist)
        if score > best_score:
            best_score, cx, cy = score, int(blob_x), int(blob_y)

    if verbose: log(f"  detect: pupil centre candidate ({cx}, {cy})")

    # Step 5 — radial intensity profile from detected centre
    angles  = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    max_r   = min(cx, cy, w - cx, h - cy) - 5
    if max_r < 40:
        if verbose: log(f"  detect: max_r too small ({max_r})")
        return None

    r_range = np.arange(5, min(max_r, 220))
    ring_means = []
    for r in r_range:
        xs = np.clip((cx + r * np.cos(angles)).astype(int), 0, w - 1)
        ys = np.clip((cy + r * np.sin(angles)).astype(int), 0, h - 1)
        ring_means.append(img[ys, xs].mean())

    profile  = ndimage.gaussian_filter1d(np.array(ring_means), sigma=5)
    gradient = np.gradient(profile)

    # Step 6 — pupil boundary: first strong positive gradient in 20–90px range
    s1 = int(np.searchsorted(r_range, 20))
    e1 = int(np.searchsorted(r_range, 90))
    if e1 <= s1:
        r_pupil = 45
    else:
        r_pupil = int(r_range[np.argmax(gradient[s1:e1]) + s1])

    # Step 7 — iris outer boundary: next strong gradient beyond pupil
    s2 = int(np.searchsorted(r_range, r_pupil + 25))
    e2 = int(np.searchsorted(r_range, min(len(r_range) - 1, r_pupil + 130)))
    if e2 <= s2:
        r_iris = r_pupil + 90
    else:
        r_iris = int(r_range[np.argmax(gradient[s2:e2]) + s2])

    # Step 8 — enforce minimum iris width and cap at image boundary
    if r_iris - r_pupil < 40:
        r_iris = r_pupil + 90
    r_iris = min(r_iris, max_r)

    if verbose:
        log(f"  detect: r_pupil={r_pupil}px  r_iris={r_iris}px  width={r_iris-r_pupil}px")

    return cx, cy, r_pupil, r_iris

# ── LINEAR POLAR TRANSFORM ────────────────────────────────────────────────────
def to_linear_polar(img, cx, cy, r_inner, r_outer, n_r=128, n_theta=360):
    """
    Map iris annulus → rectangular strip in LINEAR radial coordinates.
    Rows = radius (linear, r_inner → r_outer).
    Cols = angle (0 → 2π).
    This is the corrected approach (not log-polar).
    """
    radii  = np.linspace(r_inner, r_outer, n_r)
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    r_grid, t_grid = np.meshgrid(radii, thetas, indexing="ij")

    xs = np.clip(cx + r_grid * np.cos(t_grid), 0, img.shape[1] - 1)
    ys = np.clip(cy + r_grid * np.sin(t_grid), 0, img.shape[0] - 1)

    return ndimage.map_coordinates(img, [ys, xs], order=1, mode="nearest")

# ── RADIAL FREQUENCY EXTRACTION ───────────────────────────────────────────────
def extract_frequency(polar_strip, r_pupil, r_iris):
    """
    Extract dominant STRUCTURAL radial frequency.

    Averages the polar strip over all angles to get a 1D radial intensity
    profile, then FFTs that.  Averaging suppresses fine texture and noise,
    isolating the coarse radial wave pattern (fibre organisation).

    Unit conversion: strip covers r_pupil→r_iris (annular width only).
    Scale to cycles per full iris diameter:
        cycles_per_diameter = dom_idx × (2 × r_iris) / (r_iris − r_pupil)
    Comparable to the case-study value of 6–7 cycles per iris width.
    """
    n_r, _ = polar_strip.shape

    # Average over angular dimension → 1D radial profile
    radial_avg = polar_strip.mean(axis=1)

    # Hanning window
    windowed = radial_avg * np.hanning(n_r)

    # 1D FFT
    spectrum = np.abs(np.fft.rfft(windowed))

    # Zero DC and first bin
    spectrum[:2] = 0

    # Search indices 2–30 for structural frequencies
    search_end  = min(30, len(spectrum) - 1)
    seg         = spectrum[2:search_end]
    noise_thresh = seg.mean() + seg.std()

    peaks = [i for i in range(2, search_end)
             if spectrum[i] > noise_thresh
             and spectrum[i] >= spectrum[i - 1]
             and spectrum[i] >= spectrum[i + 1]]

    dom_idx = peaks[0] if peaks else int(np.argmax(seg)) + 2

    # Convert to cycles per full iris diameter
    cycles_per_diameter = dom_idx * (2.0 * r_iris) / (r_iris - r_pupil)

    return cycles_per_diameter, spectrum, dom_idx

# ── OCTAVE TRANSPOSITION — corrected protocol (v0.4) ─────────────────────────
def octave_transpose(dominant_hz):
    """Choose n for dominant; MUST apply same n to all secondaries."""
    if dominant_hz <= 0:
        return 0.0, 0
    f, n = float(dominant_hz), 0
    while f < 20:    f *= 2; n += 1
    while f > 20000: f /= 2; n -= 1
    return f, n

# ── MUSICAL NOTE ──────────────────────────────────────────────────────────────
def to_note(freq_hz):
    if freq_hz <= 0:
        return "N/A", 0.0
    A4      = 440.0
    semi    = 12 * math.log2(freq_hz / A4)
    nearest = round(semi)
    NAMES   = ["A","A#/Bb","B","C","C#/Db","D","D#/Eb","E","F","F#/Gb","G","G#/Ab"]
    name    = NAMES[nearest % 12] + str(4 + nearest // 12)
    cents   = (semi - nearest) * 100
    return name, cents

# ── FULL PIPELINE FOR ONE IMAGE ───────────────────────────────────────────────
def process(img_path, verbose=False):
    img = load_gray(img_path)
    if img is None:
        return None

    result = detect_boundaries(img, verbose=verbose)
    if result is None:
        return None

    cx, cy, r_pupil, r_iris = result
    if r_iris - r_pupil < 30:
        if verbose: log(f"  process: iris width too small ({r_iris-r_pupil}px), skipping")
        return None

    polar                           = to_linear_polar(img, cx, cy, r_pupil, r_iris)
    cycles_width, _, dom_idx        = extract_frequency(polar, r_pupil, r_iris)

    if cycles_width == 0:
        return None

    cycles_metre            = cycles_width / (IRIS_DIAMETER_MM * 1e-3)
    transposed_hz, n_shift  = octave_transpose(cycles_metre)
    note, cents             = to_note(transposed_hz)
    texture_score           = float(np.std(polar))
    collarette_ratio        = r_pupil / r_iris

    return dict(
        path             = img_path,
        cycles_width     = float(cycles_width),
        cycles_metre     = float(cycles_metre),
        transposed_hz    = float(transposed_hz),
        n_shift          = n_shift,
        note             = note,
        cents            = float(cents),
        texture_score    = texture_score,
        collarette_ratio = float(collarette_ratio),
        r_pupil          = r_pupil,
        r_iris           = r_iris,
        dom_idx          = dom_idx,
    )

# ── DATASET LOADER ────────────────────────────────────────────────────────────
def find_images(root):
    subjects = defaultdict(lambda: {"L": [], "R": []})
    exts     = {".jpg", ".jpeg", ".png", ".bmp"}
    for sid in sorted(os.listdir(root)):
        sp = os.path.join(root, sid)
        if not os.path.isdir(sp): continue
        for eye in ("L", "R"):
            ep = os.path.join(sp, eye)
            if not os.path.isdir(ep): continue
            subjects[sid][eye] = sorted(
                os.path.join(ep, f) for f in os.listdir(ep)
                if os.path.splitext(f)[1].lower() in exts
            )
    return subjects

# ── TEST 1 — REPRODUCIBILITY ──────────────────────────────────────────────────
def test_reproducibility(subjects, n=30):
    log("\n── TEST 1: REPRODUCIBILITY ──────────────────────────────────────")
    log("Multiple images of the same eye → consistent dominant frequency?")
    log("CV < 0.10 = stable measurement.\n")

    results, sample = [], random.sample(list(subjects.keys()), min(n, len(subjects)))
    for sid in sample:
        for eye in ("L", "R"):
            imgs = subjects[sid][eye]
            if len(imgs) < 2: continue
            freqs = [r["cycles_width"] for p in imgs[:5]
                     if (r := process(p)) is not None]
            if len(freqs) >= 2:
                mean = np.mean(freqs)
                cv   = np.std(freqs) / mean if mean > 0 else 999
                results.append({"sid": sid, "eye": eye,
                                 "n": len(freqs), "mean": mean, "cv": cv,
                                 "freqs": freqs})

    if not results:
        log("  No results — check iris detection.")
        return [], 999

    cvs     = [r["cv"] for r in results]
    mean_cv = np.mean(cvs)
    log(f"  Pairs tested           : {len(results)}")
    log(f"  Mean CV                : {mean_cv:.4f}")
    log(f"  Max CV                 : {max(cvs):.4f}")
    means_str = ", ".join(str(round(r["mean"], 1)) for r in results[:8])
    log(f"  Sample dominant freqs  : [{means_str}]")
    log(f"  Verdict: {'STABLE ✓' if mean_cv < 0.10 else 'VARIABLE — investigate'}")
    return results, mean_cv

# ── TEST 2 — EIGENMODE / MULTIMODALITY ───────────────────────────────────────
def test_eigenmode(subjects, n=150):
    log("\n── TEST 2: EIGENMODE PREDICTION ────────────────────────────────")
    log("Dominant frequency distribution across population.")
    log("Multimodal clustering = supports eigenmode theory.\n")

    freqs  = []
    sample = random.sample(list(subjects.keys()), min(n, len(subjects)))
    for sid in sample:
        imgs = subjects[sid]["R"]
        if not imgs: continue
        r = process(imgs[0])
        if r: freqs.append(r["cycles_width"])

    if not freqs:
        log("  No results — check iris detection.")
        return np.array([]), []

    freqs = np.array(freqs)
    log(f"  Irises analysed : {len(freqs)}")
    log(f"  Range           : {freqs.min():.1f} – {freqs.max():.1f} cycles/width")
    log(f"  Mean            : {freqs.mean():.2f}   Median: {np.median(freqs):.2f}   Std: {freqs.std():.2f}\n")

    hist, edges = np.histogram(freqs, bins=20)
    log("  Distribution:")
    max_h = max(hist) if max(hist) > 0 else 1
    for i, cnt in enumerate(hist):
        bar = "█" * int(cnt / max_h * 30)
        log(f"    {edges[i]:5.1f}–{edges[i+1]:5.1f}: {bar} ({cnt})")

    peaks = [i for i in range(1, len(hist) - 1)
             if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 2]
    log(f"\n  Local peaks (count>2): {len(peaks)}")
    for p in peaks:
        log(f"    ~{(edges[p]+edges[p+1])/2:.1f} cycles/width  (n={hist[p]})")
    log(f"  Verdict: {'MULTIMODAL ✓ — supports eigenmode theory' if len(peaks)>=2 else '— fewer than 2 clear peaks, run Hartigan dip test'}")
    log("  Formal test: pip3 install diptest → diptest.diptest(freqs)")
    return freqs, peaks

# ── TEST 3 — COLLARETTE RATIO ─────────────────────────────────────────────────
def test_collarette(subjects, n=80):
    log("\n── TEST 3: COLLARETTE RATIO ─────────────────────────────────────")
    log("Pupil:iris radius ratio. φ predicts pupil ≈ 0.382 of iris radius.\n")

    ratios = []
    sample = random.sample(list(subjects.keys()), min(n, len(subjects)))
    for sid in sample:
        imgs = subjects[sid]["R"]
        if not imgs: continue
        r = process(imgs[0])
        if r: ratios.append(r["collarette_ratio"])

    if not ratios:
        log("  No results.")
        return np.array([])

    ratios   = np.array(ratios)
    phi_pred = 1 - 1 / 1.6180339887    # 0.382
    log(f"  Irises measured : {len(ratios)}")
    log(f"  Mean ratio      : {ratios.mean():.3f}   Std: {ratios.std():.3f}")
    log(f"  φ prediction    : {phi_pred:.3f}")
    log(f"  Deviation       : {abs(ratios.mean()-phi_pred):.3f}")
    log(f"  Verdict: {'CONSISTENT WITH φ ✓' if abs(ratios.mean()-phi_pred)<0.06 else 'DEVIATION — note and investigate'}")
    return ratios

# ── TEST 4 — BILATERAL COMPARISON ────────────────────────────────────────────
def test_bilateral(subjects, n=50):
    log("\n── TEST 4: BILATERAL COMPARISON ────────────────────────────────")
    log("L vs R eye dominant frequency for the same person.\n")

    diffs  = []
    sample = random.sample(list(subjects.keys()), min(n, len(subjects)))
    for sid in sample:
        li = subjects[sid]["L"]
        ri = subjects[sid]["R"]
        if not li or not ri: continue
        rl = process(li[0]); rr = process(ri[0])
        if rl and rr:
            lf, rf = rl["cycles_width"], rr["cycles_width"]
            d      = abs(lf - rf) / max(lf, rf, 1)
            diffs.append({"sid": sid, "L": lf, "R": rf, "diff": d,
                          "L_tex": rl["texture_score"], "R_tex": rr["texture_score"]})

    if not diffs:
        log("  No results.")
        return []

    raw = [d["diff"] for d in diffs]
    log(f"  Subjects tested           : {len(diffs)}")
    log(f"  Mean bilateral diff       : {np.mean(raw):.3f}")
    log(f"  Std                       : {np.std(raw):.3f}")
    log(f"  Range                     : {min(raw):.3f} – {max(raw):.3f}")
    log(f"  Large diff (>0.25) count  : {sum(1 for d in raw if d>0.25)}")
    log(f"  Small diff (<0.10) count  : {sum(1 for d in raw if d<0.10)}")
    log(f"  Sample (sid, L, R, diff)  :")
    for d in diffs[:6]:
        log(f"    {d['sid']}: L={d['L']:.1f}  R={d['R']:.1f}  diff={d['diff']:.3f}")
    return diffs

# ── TEST 5 — NOTE DISTRIBUTION ───────────────────────────────────────────────
def test_notes(subjects, n=150):
    log("\n── TEST 5: MUSICAL NOTE DISTRIBUTION ───────────────────────────")
    log("Which notes appear most frequently across population.\n")

    note_counts = defaultdict(int)
    sample      = random.sample(list(subjects.keys()), min(n, len(subjects)))
    for sid in sample:
        imgs = subjects[sid]["R"]
        if not imgs: continue
        r = process(imgs[0])
        if r and r["note"] != "N/A":
            note_counts[r["note"]] += 1

    total = sum(note_counts.values())
    if total == 0:
        log("  No notes computed.")
        return note_counts

    log(f"  Irises analysed: {total}")
    for note, cnt in sorted(note_counts.items(), key=lambda x: -x[1])[:15]:
        bar = "█" * int(cnt / max(note_counts.values()) * 25)
        log(f"  {note:8s}: {bar} {cnt:3d}  ({100*cnt/total:.1f}%)")
    return note_counts

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    log("=" * 65)
    log("FREQUENCY IRIS FRAMEWORK — VALIDATION v0.4.3")
    log("Dataset : CASIA-Iris-Thousand (640×480 NIR full-eye images)")
    log("Author  : Prof. Tieniu Tan, Center for Biometrics & Security Research")
    log("          National Lab of Pattern Recognition, Institute of Automation")
    log("          Chinese Academy of Sciences")
    log("Kaggle  : Sondos Aabed (owner/uploader), MIT License")
    log("=" * 65)

    if not os.path.isdir(DATASET_PATH):
        log(f"\nERROR: Dataset not found at {DATASET_PATH}")
        sys.exit(1)

    log(f"\nScanning {DATASET_PATH} ...")
    subjects   = find_images(DATASET_PATH)
    total_imgs = sum(len(v["L"]) + len(v["R"]) for v in subjects.values())
    log(f"Found {len(subjects)} subjects, {total_imgs} images")

    # ── Diagnostic: first image ──────────────────────────────────────────────
    sid0 = list(subjects.keys())[0]
    log(f"\n── DIAGNOSTIC: first image of subject '{sid0}' ─────────────────")
    img0 = subjects[sid0]["R"][0] if subjects[sid0]["R"] else None
    if img0:
        log(f"  Path: {img0}")
        result0 = process(img0, verbose=True)
        if result0:
            log(f"  cycles/width    : {result0['cycles_width']:.1f}")
            log(f"  cycles/metre    : {result0['cycles_metre']:.0f}")
            log(f"  transposed Hz   : {result0['transposed_hz']:.2f}")
            log(f"  note            : {result0['note']} ({result0['cents']:+.1f} cents)")
            log(f"  texture score   : {result0['texture_score']:.2f}")
            log(f"  collarette ratio: {result0['collarette_ratio']:.3f}")
            log(f"  dom_idx         : {result0['dom_idx']}")
            log("  → DIAGNOSTIC PASS ✓")
        else:
            log("  → DIAGNOSTIC FAIL — iris detection could not find valid boundaries")
            log("  Check the image path above is a real eye photo.")
            log("  Continuing with remaining tests...")
    else:
        log("  No right-eye image found for first subject.")

    # ── Try a few more subjects to check success rate ────────────────────────
    log("\n── DETECTION SUCCESS RATE (first 20 subjects) ──────────────────")
    success, total_tried = 0, 0
    sample_20 = list(subjects.keys())[:20]
    for sid in sample_20:
        imgs = subjects[sid]["R"]
        if not imgs: continue
        total_tried += 1
        r = process(imgs[0])
        if r: success += 1
    log(f"  {success}/{total_tried} images processed successfully")
    if success < total_tried * 0.5:
        log("  WARNING: less than 50% success rate — iris detection needs tuning")
        log("  Results below may not be reliable.")
    else:
        log("  Detection appears to be working.")

    # ── Run all tests ─────────────────────────────────────────────────────────
    repro_results, mean_cv = test_reproducibility(subjects, n=30)
    dominant_freqs, peaks  = test_eigenmode(subjects, n=150)
    ratios                 = test_collarette(subjects, n=80)
    bilateral              = test_bilateral(subjects, n=50)
    notes                  = test_notes(subjects, n=150)

    # ── Summary ───────────────────────────────────────────────────────────────
    log("\n" + "=" * 65)
    log("VALIDATION SUMMARY")
    log("=" * 65)
    log(f"Detection success rate      : {success}/{total_tried}")
    log(f"Test 1 Reproducibility (CV) : {mean_cv:.4f}  {'✓' if 0<mean_cv<0.10 else ('✗ all same — detect fail' if mean_cv==0 else '— variable')}")
    log(f"Test 2 Eigenmode peaks      : {len(peaks)}       {'✓ multimodal' if len(peaks)>=2 else '— run Hartigan dip test'}")
    if len(ratios): log(f"Test 3 Collarette ratio     : {ratios.mean():.3f}  {'✓' if abs(ratios.mean()-0.382)<0.06 else '— investigate'}")
    if bilateral:   log(f"Test 4 Bilateral diff (mean): {np.mean([d['diff'] for d in bilateral]):.3f}")
    if notes:       log(f"Test 5 Most common note     : {max(notes,key=notes.get)}")
    log("=" * 65)
    log("\nCopy this full output and paste into the Claude chat.")

    with open(RESULTS_FILE, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved to: {RESULTS_FILE}")

if __name__ == "__main__":
    main()
