from matplotlib import pyplot
import numpy
import pymc3
import scipy
import scipy.stats
import seaborn


class BinaryModel:
    """
    alpha = successes
    beta = failures
    """
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.total = alpha + beta
        self.odds = float(alpha)/self.total
        self.obs = scipy.stats.bernoulli.rvs(self.odds, size=self.total)


def run_mc(a, b, n_samples):
    with pymc3.Model() as model:
        theta_a = pymc3.Beta('theta_a', alpha=1 + a.alpha, beta=1 + a.beta)  # prior
        theta_b = pymc3.Beta('theta_b', alpha=1 + b.alpha, beta=1 + b.beta)  # prior

        data_a = pymc3.Bernoulli('observed_A', p=theta_a, observed=a.obs)
        data_b = pymc3.Bernoulli('observed B', p=theta_b, observed=b.obs)
        delta = pymc3.Deterministic("delta", theta_a - theta_b)

        start = pymc3.find_MAP()  # Find good starting point
        step = pymc3.Metropolis()  # MCMC sampling algorithm
        trace = pymc3.sample(n_samples, step, start=start)

#    pymc3.traceplot(trace)
#    pyplot.show()
    return trace[1000:]  # burning first few traces for more consistent results


def plot_post_dist(trace):
    """
    Plots posterior distributions
    :param trace:
    :return:
    """
    pyplot.figure()
    seaborn.distplot(trace['theta_a'],  label="theta_a")
    seaborn.distplot(trace['theta_b'],  label="theta_b")
    pyplot.xlabel("Probability")
    pyplot.ylabel("Frequency")
    pyplot.title("Posterior Distribution")
    pyplot.legend()


def plot_joint_dist(trace):
    """
    Plots Joint Probability distribution
    :param trace:
    :return:
    """
    pyplot.figure()
    joint = seaborn.kdeplot(trace['theta_a'], trace['theta_b'], shade=True)
    y_min, y_max = joint.get_ylim()
    x_min, x_max = joint.get_xlim()
    trace_range = numpy.linspace(max(x_min, y_min), min(x_max, y_max))
    pyplot.plot(trace_range, trace_range)
    pyplot.xlabel("theta a")
    pyplot.ylabel("theta b")
    pyplot.title("Joint Probability")


def value_dist(trace):
    """
    Calculates Probability and value remaining data
    :param trace:
    :return:
    """
    p_b_beats_a =  numpy.mean(trace['delta'] < 0)
    numpy.mean(trace['theta_a'] < trace['theta_b'])

    if p_b_beats_a > 0.5:
        best_page = "B"
        value_remaining = numpy.where(trace['theta_a'] < trace['theta_b'], 0, (trace['theta_a'] - trace['theta_b']) / trace['theta_b'])
    else:
        best_page = "A"
        value_remaining = numpy.where(trace['theta_a'] > trace['theta_b'], 0, (trace['theta_b'] - trace['theta_a']) / trace['theta_a'])
    percentile = numpy.percentile(value_remaining, 95)
    return p_b_beats_a, best_page, percentile, value_remaining


def plot_value_dist(best_page, percentile, value_remaining):
    """
    Plots Value Remaining Distribution

    :param best_page:
    :param percentile:
    :param value_remaining:
    :return:
    """
    pyplot.figure()
    value = seaborn.distplot(value_remaining)
    _, y_max = value.get_ylim()
    line = numpy.linspace(0, y_max)
    pyplot.plot([percentile for _ in range(len(line))], line, label="95th Percentile")
    pyplot.title("Value Remaining of {}".format(best_page))
    pyplot.xlabel("Probability")
    pyplot.ylabel("Frequency")
    pyplot.legend()


def print_info(p_b_beats_a, percentile, threshold=0.01):
    """
    Prints Info

    :param p_b_beats_a:
    :param percentile:
    :param threshold:
    :return:
    """
    print "\n"
    print 'Probability that page B is better than A = {:0.4f}'.format(p_b_beats_a)
    print "Value Remaining of {}".format(best_page)
    print "{:0.2f}% chance A is better than B".format(percentile * 100)
    if percentile < threshold:
        print "Below threshold {}. Accept B and stop test".format(threshold)
    else:
        print "Above threshold {}. Continue test".format(threshold)


if __name__ == "__main__":
    a = BinaryModel(50, 50)  # success, failures
    b = BinaryModel(60, 40)

    traces = run_mc(a, b, n_samples=20000)
    p_b_beats_a, best_page, percentile, value_remaining = value_dist(traces)

    plot_post_dist(traces)
    plot_joint_dist(traces)
    plot_value_dist(best_page, percentile, value_remaining)

    print_info(p_b_beats_a, percentile, threshold=0.01)
    pyplot.show()

