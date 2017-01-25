# A/B testing with Bayesian with stopping
- pymc3 A/B testing for webpage counts
- calculates P(B > A) and 95th confidence interval
- generates plots for visualization of A/B test
- Implementing Google Analytics multi-armed bandit stopping method (https://support.google.com/analytics/answer/2844870?hl=en)

## Getting started
Change the values to generate plots

- a = BinaryModel(50, 50)  # success, failures
- b = BinaryModel(60, 40)

    pip install -r requirements
    python bayes.py
    
