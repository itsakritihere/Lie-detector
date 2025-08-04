def predict_truth_or_lie_from_features(features):
    feature_names = [
        "pause_count", "longest_pause", "pause_rate", "pitch_jump_count", "pitch_range",
        "pitch_skewness", "pitch_std", "mfcc13_microstress", "flux_spikes", "energy_range",
        "pitch_entropy", "spectral_rolloff"
    ]

    # Convert list to dict if needed
    if isinstance(features, (list, tuple)):
        features = dict(zip(feature_names, features))

    def fuzzy_range(value, low, mid, high, weight):
        if value <= low:
            return 1 * weight
        elif value >= high:
            return 0
        elif low < value < mid:
            return ((mid - value) / (mid - low)) * weight
        elif mid <= value < high:
            return ((high - value) / (high - mid)) * weight
        else:
            return 0

    # Weights for features reflecting their importance
    weights = {
        "pause_count": 8,        # Increased weight for pause_count
        "longest_pause": 6,
        "pause_rate": 6,
        "pitch_jump_count": 5,
        "pitch_range": 5,
        "pitch_skewness": 4,
        "pitch_std": 5,
        "mfcc13_microstress": 5,
        "flux_spikes": 4,
        "energy_range": 4,
        "pitch_entropy": 4,
        "spectral_rolloff": 4
    }

    score = 0

    # Pause features: lower is better (more truthful)
    score += fuzzy_range(features["pause_count"], 0, 10, 30, weights["pause_count"])
    score += fuzzy_range(features["longest_pause"], 0, 0.02, 0.12, weights["longest_pause"])
    score += fuzzy_range(features["pause_rate"], 0, 7, 15, weights["pause_rate"])

    # Pitch-related features: mid-range or moderate-high values favored
    score += fuzzy_range(features["pitch_jump_count"], 10, 25, 50, weights["pitch_jump_count"])
    score += fuzzy_range(features["pitch_range"], 40, 90, 150, weights["pitch_range"])

    # Pitch skewness: full weight if between 0.5 and 1.2, else linearly decreasing
    skew = features["pitch_skewness"]
    if 0.5 <= skew <= 1.2:
        score += weights["pitch_skewness"]
    else:
        # Linear penalty for skew outside [0.5, 1.2]
        penalty = max(0, 1 - abs(skew - 0.85) / 2)
        score += weights["pitch_skewness"] * penalty

    score += fuzzy_range(features["pitch_std"], 40, 80, 140, weights["pitch_std"])

    # Microstress and spectral flux (moderate values better)
    score += fuzzy_range(features["mfcc13_microstress"], 25, 40, 60, weights["mfcc13_microstress"])
    score += fuzzy_range(features["flux_spikes"], 20, 50, 90, weights["flux_spikes"])
    score += fuzzy_range(features["energy_range"], 0.1, 0.4, 0.8, weights["energy_range"])

    # Spectral entropy and rolloff: mid-range better
    score += fuzzy_range(features["pitch_entropy"], 0.8, 1.4, 1.8, weights["pitch_entropy"])
    score += fuzzy_range(features["spectral_rolloff"], 3000, 4800, 7600, weights["spectral_rolloff"])

    total_weight = sum(weights.values())
    truth_score = (score / total_weight) * 100

    # Lowered threshold to 48% for better truth capture
    prediction = "Truth" if truth_score >= 46 else "Lie"

    return prediction, round(truth_score, 2)
