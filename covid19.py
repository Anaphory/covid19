""" MCMC of COVID-19 epidemiologics using a compartment (SIR) model

"""

import csv
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy
from numpy.random import randint as rand
from numpy import array, cumsum
from scipy.stats import binom, beta
from country_data import population_sizes

limit = ["Switzerland", "Italy", "Germany",
         "Netherlands", "Austria", "France",
         "Belgium", "Luxembourg", "Denmark"]
countries = []

# Data
d = []
with open("COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv") as data:
    for row in csv.reader(data):
        if row[3] == "Long":
            continue
        if row[1] in limit and row[0].strip() in ["", row[1]]:
            d.append([0] * 200 + [int(i) for i in row[4:]])
            countries.append(row[1])
cum_tested_positive = array(d)

d = []
with open("COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv") as data:
    for row in csv.reader(data):
        if row[3] == "Long":
            continue
        if row[1] in limit and row[0].strip() in ["", row[1]]:
            d.append([0] * 200 + [int(i) for i in row[4:]])
cum_deaths = array(d)

d = []
with open("COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv") as data:
    for row in csv.reader(data):
        if row[3] == "Long":
            continue
        if row[1] in limit and row[0].strip() in ["", row[1]]:
            d.append([0] * 200 + [int(i) for i in row[4:]])
cum_observed_recoveries = array(d)

days = cum_deaths.shape[1]
populations = array([population_sizes.get(country, 10_000_000)
    for country in countries])[:, None]

tested_positive = cum_tested_positive[:, 1:] - cum_tested_positive[:, :-1]
deaths = cum_deaths[:, 1:] - cum_deaths[:, :-1]
observed_recoveries = numpy.maximum(
    cum_observed_recoveries[:, 1:] - cum_observed_recoveries[:, :-1], 0)

# Parameters
def log_lk(newly_exposed, newly_infected, unobserved,
           exposed_p, external_sources_p, infected_p, tested_p, tested_contact_p, dead_p, immune_p, susceptible_p,
           dead_alpha, dead_beta, test_alpha, test_beta, contact_alpha, contact_beta):
    if (exposed_p<=0 or external_sources_p<=0 or infected_p<=0 or (tested_p<=0).any() or (tested_contact_p<=0).any() or (dead_p<=0).any() or immune_p<=0 or susceptible_p<=0 or
        # exposed_p>=1 or
        external_sources_p>=1 or infected_p>=1 or (tested_p>=1).any() or (tested_contact_p>=1).any() or (dead_p>=1).any() or immune_p>=1 or susceptible_p>=1):
        return -numpy.inf, numpy.inf
    cum_exposed = numpy.hstack(
        (numpy.zeros_like(populations),
         cumsum(newly_exposed - newly_infected, axis=1)))
    cum_unobserved = numpy.hstack(
        (numpy.zeros_like(populations),
         cumsum(unobserved, axis=1)))
    cum_unknown_infected = numpy.hstack(
        (numpy.zeros_like(populations),
         cumsum(newly_infected - unobserved - tested_positive, axis=1)))
    cum_susceptible = populations - cum_exposed - cum_unknown_infected - cum_tested_positive - cum_deaths - cum_observed_recoveries - cum_unobserved
    log_lk = numpy.vstack((
        # Susceptible people get exposed to infected or tested
        binom.logpmf(newly_exposed,
                    cum_susceptible[:, :-1],
                    exposed_p * (
                        cum_unknown_infected[:, :-1] +
                        tested_contact_p[:, None] * cum_tested_positive[:, :-1]) +
                     external_sources_p),
        # Exposed people become infected
        binom.logpmf(newly_infected,
                     cum_exposed[:, :-1],
                     infected_p),
        # Infected people become tested
        binom.logpmf(tested_positive,
                     cum_unknown_infected[:, :-1],
                     tested_p[:, None]),
        # Or they disappear otherwise, becoming immune or dead before they are tested
        binom.logpmf(unobserved,
                     cum_unknown_infected[:, :-1],
                     dead_p[:, None] + immune_p),
        # Tested people recover
        binom.logpmf(observed_recoveries,
                     cum_tested_positive[:, :-1],
                     immune_p),
        # or they die
        binom.logpmf(deaths,
                     cum_tested_positive[:, :-1],
                     dead_p[:, None]),
    ))
    errors = numpy.isfinite(log_lk)
    log_lk[~errors] = 0
    prior = (
        beta.logpdf(dead_p, dead_alpha, dead_beta).sum(),
        beta.logpdf(tested_p, test_alpha, test_beta).sum(),
        beta.logpdf(tested_contact_p, contact_alpha, contact_beta).sum(),
    )
    return numpy.sum(log_lk) + sum(prior), errors.sum() + (array((newly_exposed, newly_infected, unobserved)) < 0).sum()

state = dict(
    newly_exposed = numpy.zeros_like(tested_positive),
    newly_infected = numpy.zeros_like(tested_positive),
    unobserved = numpy.zeros_like(tested_positive),
    exposed_p = 0.05,
    infected_p = 0.5,
    external_sources_p = 1e-10,
    tested_p = numpy.ones(len(populations)) * 0.5,
    tested_contact_p = numpy.ones(len(populations)) * 0.5,
    dead_p = numpy.ones(len(populations)) * 0.5,
    immune_p = 0.5,
    dead_alpha = 2,
    dead_beta = 2,
    test_alpha = 2,
    test_beta = 2,
    contact_alpha = 2,
    contact_beta = 2,
    susceptible_p = 0.5)

def propose(state):
    i = rand(23)
    if i < 3:
        index = tuple(rand(state["newly_exposed"].shape))
        state["newly_exposed"][index] += rand(1, 3) * (-1) ** rand(2)
        state["newly_exposed"][state["newly_exposed"] < 0] = 0
    elif i < 6:
        index = tuple(rand(state["newly_infected"].shape))
        state["newly_infected"][index] += rand(1, 3) * (-1) ** rand(2)
        state["newly_infected"][state["newly_infected"] < 0] = 0
    elif i < 9:
        index = tuple(rand(state["unobserved"].shape))
        state["unobserved"][index] += rand(1, 3) * (-1) ** rand(2)
        state["unobserved"][state["unobserved"] < 0] = 0
    elif i < 10:
        state["exposed_p"] = 0.01 * (numpy.random.random() - 0.5) + state["exposed_p"]
    elif i < 11:
        state["infected_p"] = 0.01 * (numpy.random.random() - 0.5) + state["infected_p"]
    elif i < 12:
        index = tuple(rand(state["tested_p"].shape))
        state["tested_p"][index] = 0.01 * (numpy.random.random() - 0.5) + state["tested_p"][index]
    elif i < 13:
        index = tuple(rand(state["dead_p"].shape))
        state["dead_p"][index] = 0.01 * (numpy.random.random() - 0.5) + state["dead_p"][index]
    elif i < 14:
        state["immune_p"] = 0.01 * (numpy.random.random() - 0.5) + state["immune_p"]
    elif i < 15:
        state["susceptible_p"] = 0.01 * (numpy.random.random() - 0.5) + state["susceptible_p"]
    elif i < 17:
        index = tuple(rand(state["tested_contact_p"].shape))
        state["tested_contact_p"][index] = 0.01 * (numpy.random.random() - 0.5) + state["susceptible_p"]
    elif i < 18:
        state["dead_alpha"] += rand(1, 3) * (-1) ** rand(2)
    elif i < 19:
        state["dead_beta"] += rand(1, 3) * (-1) ** rand(2)
    elif i < 20:
        state["test_alpha"] += rand(1, 3) * (-1) ** rand(2)
    elif i < 21:
        state["test_beta"] += rand(1, 3) * (-1) ** rand(2)
    elif i < 22:
        state["contact_alpha"] += rand(1, 3) * (-1) ** rand(2)
    elif i < 23:
        state["contact_beta"] += rand(1, 3) * (-1) ** rand(2)
    elif i < 11:
        state["external_sources_p"] = 1e-10 * (numpy.random.random() - 0.5) + state["external_sources_p"]

print(*
      ["s", "lk", "err", "exposed_p", "external_sources_p", "infected_p", "immune_p"] +
      ["tested_p {:}".format(c) for c in countries] +
      ["tested_contact_p {:}".format(c) for c in countries] +
      ["dead_p {:}".format(c) for c in countries],
      sep=", ")
lk, err = log_lk(**state)
for s in range(2000000):
    if s % 10000 == 0:
        print(*[s, lk, err, state["exposed_p"], state["external_sources_p"], state["infected_p"], state["immune_p"]] +
              list(state["tested_p"]) +
              list(state["tested_contact_p"]) +
              list(state["dead_p"]),
              sep=", ")
    copy_state = deepcopy(state)
    propose(copy_state)
    new_lk, new_err = log_lk(**copy_state)
    if new_err < err or (new_err == err and new_lk > lk):
        state = copy_state
        lk = new_lk
        err = new_err
    elif new_err < err + 1 and numpy.random.random() < numpy.exp(new_lk - lk):
        state = copy_state
        lk = new_lk
        err = new_err

