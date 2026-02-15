//! Statistical methods for Elo estimation, SPRT, and pentanomial model.

use crate::types::{PentanomialCounts, SprtDecision};

/// Convert Elo difference to expected score (logistic model).
/// `score = 1 / (1 + 10^(-elo/400))`
pub fn score_from_elo(elo: f64) -> f64 {
    1.0 / (1.0 + f64::powf(10.0, -elo / 400.0))
}

/// Convert expected score to Elo difference (logistic model).
/// `elo = -400 * log10(1/s - 1)`
pub fn elo_from_score(score: f64) -> f64 {
    if score <= 0.0 || score >= 1.0 {
        if score <= 0.0 {
            return f64::NEG_INFINITY;
        }
        return f64::INFINITY;
    }
    -400.0 * f64::log10(1.0 / score - 1.0)
}

/// Elo estimate with 95% confidence interval.
#[derive(Debug, Clone, Copy)]
pub struct EloEstimate {
    pub elo: f64,
    pub error_margin: f64,
    pub los: f64,
}

impl EloEstimate {
    pub fn lower(&self) -> f64 {
        self.elo - self.error_margin
    }

    pub fn upper(&self) -> f64 {
        self.elo + self.error_margin
    }
}

/// Compute Elo estimate from pentanomial counts.
/// Uses the pentanomial model for tighter confidence intervals.
pub fn elo_estimate_pentanomial(penta: &PentanomialCounts) -> Option<EloEstimate> {
    let n = penta.total_pairs() as f64;
    if n < 2.0 {
        return None;
    }

    // Pair scores: 0/2, 1/2, 2/2, 3/2, 4/2 = 0, 0.25, 0.5, 0.75, 1.0
    let pair_scores = [0.0, 0.25, 0.5, 0.75, 1.0];

    // Mean pair score
    let mut mean = 0.0;
    for (i, &count) in penta.counts.iter().enumerate() {
        mean += pair_scores[i] * count as f64;
    }
    mean /= n;

    // Variance of pair score
    let mut var = 0.0;
    for (i, &count) in penta.counts.iter().enumerate() {
        let diff = pair_scores[i] - mean;
        var += diff * diff * count as f64;
    }
    var /= n - 1.0;

    if var <= 0.0 {
        // All games identical — Elo is 0 with 0 error.
        return Some(EloEstimate {
            elo: elo_from_score(mean),
            error_margin: 0.0,
            los: if mean > 0.5 {
                1.0
            } else if mean < 0.5 {
                0.0
            } else {
                0.5
            },
        });
    }

    let stdev = var.sqrt();
    let se = stdev / n.sqrt();

    let elo = elo_from_score(mean);

    // 95% CI: mean ± 1.96 * SE, convert to Elo
    let lower = elo_from_score((mean - 1.96 * se).clamp(0.001, 0.999));
    let upper = elo_from_score((mean + 1.96 * se).clamp(0.001, 0.999));
    let error_margin = (upper - lower) / 2.0;

    // LOS (Likelihood of Superiority): P(score > 0.5)
    let los = normal_cdf((mean - 0.5) / se);

    Some(EloEstimate {
        elo,
        error_margin,
        los,
    })
}

/// Compute Elo estimate from simple W/D/L counts.
pub fn elo_estimate_trinomial(wins: u64, draws: u64, losses: u64) -> Option<EloEstimate> {
    let n = (wins + draws + losses) as f64;
    if n < 2.0 {
        return None;
    }

    let w = wins as f64 / n;
    let d = draws as f64 / n;
    let score = w + d * 0.5;

    // Variance of trinomial score: E[X^2] - E[X]^2
    // Where X = 1 with prob w, 0.5 with prob d, 0 with prob l
    let var = (w + d * 0.25) - score * score;
    let var_corrected = var * n / (n - 1.0);

    if var_corrected <= 0.0 {
        return Some(EloEstimate {
            elo: elo_from_score(score),
            error_margin: 0.0,
            los: if score > 0.5 {
                1.0
            } else if score < 0.5 {
                0.0
            } else {
                0.5
            },
        });
    }

    let stdev = var_corrected.sqrt();
    let se = stdev / n.sqrt();

    let elo = elo_from_score(score);
    let lower = elo_from_score((score - 1.96 * se).clamp(0.001, 0.999));
    let upper = elo_from_score((score + 1.96 * se).clamp(0.001, 0.999));
    let error_margin = (upper - lower) / 2.0;

    let los = normal_cdf((score - 0.5) / se);

    Some(EloEstimate {
        elo,
        error_margin,
        los,
    })
}

/// SPRT (Sequential Probability Ratio Test) using the pentanomial model.
///
/// Returns the log-likelihood ratio (LLR) for testing H0: Elo <= elo0 vs H1: Elo >= elo1.
///
/// Uses the Fishtest LLR_alt2 formula:
///   LLR = N * (s1 - s0) * (2*mu - s0 - s1) / (2 * variance)
///
/// where s0 = score_from_elo(elo0), s1 = score_from_elo(elo1), mu = observed mean score,
/// and variance is the observed sample variance.
pub fn sprt_llr(penta: &PentanomialCounts, elo0: f64, elo1: f64) -> f64 {
    let n = penta.total_pairs() as f64;
    if n < 1.0 {
        return 0.0;
    }

    let s0 = score_from_elo(elo0);
    let s1 = score_from_elo(elo1);

    // Pair scores: 0, 0.25, 0.5, 0.75, 1.0
    let pair_scores = [0.0, 0.25, 0.5, 0.75, 1.0];

    // Mean
    let mut mu = 0.0;
    for (i, &count) in penta.counts.iter().enumerate() {
        mu += pair_scores[i] * count as f64;
    }
    mu /= n;

    // Variance (sample variance)
    let mut var = 0.0;
    for (i, &count) in penta.counts.iter().enumerate() {
        let diff = pair_scores[i] - mu;
        var += diff * diff * count as f64;
    }
    var /= n; // Use population variance for LLR (Fishtest convention)

    if var <= 0.0 {
        // Zero variance means all pairs have identical scores.
        // The LLR is determined by which hypothesis the observed mean is closer to.
        let mid = (s0 + s1) / 2.0;
        if mu > mid {
            return f64::INFINITY;
        } else if mu < mid {
            return f64::NEG_INFINITY;
        }
        return 0.0;
    }

    n * (s1 - s0) * (2.0 * mu - s0 - s1) / (2.0 * var)
}

/// Compute the Wald SPRT boundaries.
///
/// Returns (lower_bound, upper_bound).
/// - If LLR <= lower_bound: accept H0 (reject the patch).
/// - If LLR >= upper_bound: accept H1 (accept the patch).
pub fn sprt_bounds(alpha: f64, beta: f64) -> (f64, f64) {
    let lower = f64::ln(beta / (1.0 - alpha));
    let upper = f64::ln((1.0 - beta) / alpha);
    (lower, upper)
}

/// Minimum number of pairs before SPRT can make a decision.
/// Prevents premature termination from zero-variance edge cases
/// (e.g., one engine losing every game from the start).
pub const SPRT_MIN_PAIRS: u64 = 16;

/// Evaluate SPRT decision given current LLR, bounds, and number of pairs played.
pub fn sprt_decision(llr: f64, lower: f64, upper: f64, pairs: u64) -> SprtDecision {
    if pairs < SPRT_MIN_PAIRS {
        return SprtDecision::InProgress;
    }
    if llr >= upper {
        SprtDecision::Accept
    } else if llr <= lower {
        SprtDecision::Reject
    } else {
        SprtDecision::InProgress
    }
}

/// Standard normal CDF (cumulative distribution function).
/// Approximation using error function.
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation (Abramowitz & Stegun 7.1.26).
/// Maximum error: 1.5e-7.
fn erf(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    // Constants for the approximation
    let p = 0.3275911;
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;

    let t = 1.0 / (1.0 + p * x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    let result = 1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * f64::exp(-x * x);

    sign * result
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_score_from_elo_zero() {
        assert!((score_from_elo(0.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_score_from_elo_positive() {
        // +400 Elo should give ~0.909
        let s = score_from_elo(400.0);
        assert!((s - 10.0 / 11.0).abs() < 1e-6);
    }

    #[test]
    fn test_score_from_elo_negative() {
        // -400 Elo should give ~0.091
        let s = score_from_elo(-400.0);
        assert!((s - 1.0 / 11.0).abs() < 1e-6);
    }

    #[test]
    fn test_elo_from_score_half() {
        assert!((elo_from_score(0.5) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_elo_score_roundtrip() {
        for elo in [-200.0, -50.0, 0.0, 10.0, 50.0, 200.0] {
            let score = score_from_elo(elo);
            let recovered = elo_from_score(score);
            assert!(
                (elo - recovered).abs() < 1e-6,
                "roundtrip failed for elo={elo}: got {recovered}"
            );
        }
    }

    #[test]
    fn test_elo_from_score_boundaries() {
        assert!(elo_from_score(0.0).is_infinite() && elo_from_score(0.0) < 0.0);
        assert!(elo_from_score(1.0).is_infinite() && elo_from_score(1.0) > 0.0);
    }

    #[test]
    fn test_normal_cdf_zero() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_normal_cdf_positive() {
        // Phi(1.96) ≈ 0.975
        let v = normal_cdf(1.96);
        assert!((v - 0.975).abs() < 0.001);
    }

    #[test]
    fn test_normal_cdf_negative() {
        // Phi(-1.96) ≈ 0.025
        let v = normal_cdf(-1.96);
        assert!((v - 0.025).abs() < 0.001);
    }

    #[test]
    fn test_normal_cdf_large() {
        // Phi(3.0) ≈ 0.99865
        let v = normal_cdf(3.0);
        assert!((v - 0.99865).abs() < 0.001);
    }

    #[test]
    fn test_sprt_bounds_default() {
        let (lower, upper) = sprt_bounds(0.05, 0.05);
        // lower ≈ ln(0.05/0.95) ≈ -2.944
        // upper ≈ ln(0.95/0.05) ≈  2.944
        assert!((lower - (-2.944)).abs() < 0.01);
        assert!((upper - 2.944).abs() < 0.01);
        // They should be symmetric for equal alpha/beta
        assert!((lower + upper).abs() < 1e-10);
    }

    #[test]
    fn test_sprt_llr_equal_engines() {
        // With perfectly equal results, LLR should be near 0 or slightly negative
        // (since the expected score under H0 and H1 are both mapped differently).
        let penta = PentanomialCounts {
            counts: [10, 50, 200, 50, 10],
        };
        let llr = sprt_llr(&penta, 0.0, 5.0);
        // With perfectly symmetric results (score = 0.5), the LLR should be negative
        // because the observed mean is closer to H0 (elo=0 -> score=0.5) than H1 (elo=5 -> score>0.5).
        assert!(
            llr < 0.5,
            "LLR for equal engines should be near 0 or negative, got {llr}"
        );
    }

    #[test]
    fn test_sprt_llr_winning_engine() {
        // Engine A clearly winning → LLR should be positive
        let penta = PentanomialCounts {
            counts: [1, 10, 100, 80, 30],
        };
        let llr = sprt_llr(&penta, 0.0, 5.0);
        assert!(
            llr > 0.0,
            "LLR for winning engine should be positive, got {llr}"
        );
    }

    #[test]
    fn test_sprt_llr_losing_engine() {
        // Engine A clearly losing → LLR should be strongly negative
        let penta = PentanomialCounts {
            counts: [30, 80, 100, 10, 1],
        };
        let llr = sprt_llr(&penta, 0.0, 5.0);
        assert!(
            llr < 0.0,
            "LLR for losing engine should be negative, got {llr}"
        );
    }

    #[test]
    fn test_sprt_llr_all_losses() {
        // All pairs are LL (zero variance) → LLR should be -inf
        let penta = PentanomialCounts {
            counts: [10, 0, 0, 0, 0],
        };
        let llr = sprt_llr(&penta, 0.0, 5.0);
        assert!(
            llr.is_infinite() && llr < 0.0,
            "LLR for all-losses should be -inf, got {llr}"
        );
    }

    #[test]
    fn test_sprt_llr_all_wins() {
        // All pairs are WW (zero variance) → LLR should be +inf
        let penta = PentanomialCounts {
            counts: [0, 0, 0, 0, 10],
        };
        let llr = sprt_llr(&penta, 0.0, 5.0);
        assert!(
            llr.is_infinite() && llr > 0.0,
            "LLR for all-wins should be +inf, got {llr}"
        );
    }

    #[test]
    fn test_sprt_llr_all_draws() {
        // All pairs are DD (score = 0.5, zero variance) → LLR should be -inf
        // because score 0.5 is closer to H0 (elo=0, score=0.5) than H1 (elo=5, score>0.5)
        let penta = PentanomialCounts {
            counts: [0, 0, 10, 0, 0],
        };
        let llr = sprt_llr(&penta, 0.0, 5.0);
        assert!(
            llr.is_infinite() && llr < 0.0,
            "LLR for all-draws should be -inf, got {llr}"
        );
    }

    #[test]
    fn test_sprt_decision_below_min_pairs() {
        let (lower, upper) = sprt_bounds(0.05, 0.05);
        // Below SPRT_MIN_PAIRS, always InProgress regardless of LLR
        assert_eq!(
            sprt_decision(100.0, lower, upper, 1),
            SprtDecision::InProgress
        );
        assert_eq!(
            sprt_decision(-100.0, lower, upper, 1),
            SprtDecision::InProgress
        );
    }

    #[test]
    fn test_sprt_decision() {
        let (lower, upper) = sprt_bounds(0.05, 0.05);
        let n = SPRT_MIN_PAIRS; // enough pairs
        assert_eq!(
            sprt_decision(0.0, lower, upper, n),
            SprtDecision::InProgress
        );
        assert_eq!(sprt_decision(3.0, lower, upper, n), SprtDecision::Accept);
        assert_eq!(sprt_decision(-3.0, lower, upper, n), SprtDecision::Reject);
    }

    #[test]
    fn test_elo_estimate_pentanomial_equal() {
        let penta = PentanomialCounts {
            counts: [10, 50, 200, 50, 10],
        };
        let est = elo_estimate_pentanomial(&penta).unwrap();
        // Symmetric → Elo ≈ 0
        assert!(
            est.elo.abs() < 5.0,
            "Elo for symmetric results should be near 0, got {}",
            est.elo
        );
        // CI should include 0
        assert!(est.lower() < 0.0 && est.upper() > 0.0);
        // LOS should be near 0.5
        assert!((est.los - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_elo_estimate_pentanomial_winning() {
        let penta = PentanomialCounts {
            counts: [0, 5, 50, 30, 15],
        };
        let est = elo_estimate_pentanomial(&penta).unwrap();
        assert!(est.elo > 0.0, "Elo should be positive, got {}", est.elo);
        assert!(est.los > 0.9, "LOS should be high, got {}", est.los);
    }

    #[test]
    fn test_elo_estimate_trinomial() {
        let est = elo_estimate_trinomial(50, 200, 50).unwrap();
        assert!(est.elo.abs() < 5.0);
        assert!(est.lower() < 0.0 && est.upper() > 0.0);
    }

    #[test]
    fn test_elo_estimate_insufficient_data() {
        assert!(elo_estimate_pentanomial(&PentanomialCounts::default()).is_none());
        let penta = PentanomialCounts {
            counts: [1, 0, 0, 0, 0],
        };
        assert!(elo_estimate_pentanomial(&penta).is_none());
    }

    #[test]
    fn test_erf_symmetry() {
        // erf(-x) = -erf(x)
        for x in [0.5, 1.0, 1.5, 2.0] {
            assert!((erf(x) + erf(-x)).abs() < 1e-6);
        }
    }

    #[test]
    fn test_erf_known_values() {
        // erf(0) ≈ 0 (within approximation error)
        assert!(erf(0.0).abs() < 1e-6);
        // erf(1) ≈ 0.8427
        assert!((erf(1.0) - 0.8427).abs() < 0.001);
    }
}
