""" MCMC of COVID-19 epidemiologics using a compartment (SIR) model

"""

import csv
import sys
import json
import numpy
from copy import deepcopy
import matplotlib.pyplot as plt
from numpy import array, cumsum
from scipy.stats import binom, beta
from numpy.random import randint as rand
from country_data import population_sizes

limit = ["Switzerland", "Italy", "Germany",
         "Netherlands", "Austria", "France",
         "Belgium", "Luxembourg", "Denmark", "United Kingdom",
         "Iran", "Korea, South", "Spain"]
countries = []

# Data
d = []
with open("COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv") as data:
    for row in csv.reader(data):
        if row[3] == "Long":
            continue
        if row[1] == "China" or (row[1] in limit and row[0].strip() in ["", row[1]]):
            d.append([0] * 200 + [int(i) for i in row[4:]])
            countries.append(row[1] + "-" + row[0])
cum_confirmed = array(d)

d = []
with open("COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv") as data:
    for row in csv.reader(data):
        if row[3] == "Long":
            continue
        if row[1] == "China" or (row[1] in limit and row[0].strip() in ["", row[1]]):
            d.append([0] * 200 + [int(i) for i in row[4:]])
cum_deaths = array(d)

d = []
with open("COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv") as data:
    for row in csv.reader(data):
        if row[3] == "Long":
            continue
        if row[1] == "China" or (row[1] in limit and row[0].strip() in ["", row[1]]):
            d.append([0] * 200 + [int(i) for i in row[4:]])
cum_recovered = array(d)

days = cum_deaths.shape[1]

populations = array([population_sizes.get(country, 10_000_000)
    for country in countries])[:, None]

confirmed = cum_confirmed[:, 1:] - cum_confirmed[:, :-1]
deaths = cum_deaths[:, 1:] - cum_deaths[:, :-1]
recovered = numpy.maximum(
    cum_recovered[:, 1:] - cum_recovered[:, :-1], 0)

# Parameters
def log_lk(newly_exposed, newly_infected, unobserved,
           exposed_p, external_sources_p, infected_p, tested_p, tested_contact_p, dead_p, immune_p, susceptible_p,
           dead_alpha, dead_beta, test_alpha, test_beta, contact_alpha, contact_beta):
    if (exposed_p<=0 or external_sources_p<=0 or infected_p<=0 or (tested_p<=0).any() or (tested_contact_p<=0).any() or (dead_p<=0).any() or immune_p<=0 or susceptible_p<=0 or
        # exposed_p>=1 or
        external_sources_p>=1 or infected_p>=1 or (tested_p>=1).any() or (tested_contact_p>=1).any() or (dead_p>=1).any() or immune_p>=1 or susceptible_p>=1):
        return -numpy.inf, None
    cum_exposed = numpy.hstack(
        (numpy.zeros_like(populations),
         cumsum(newly_exposed - newly_infected, axis=1)))
    cum_unobserved = numpy.hstack(
        (numpy.zeros_like(populations),
         cumsum(unobserved, axis=1)))
    cum_unknown_infected = numpy.hstack(
        (numpy.zeros_like(populations),
         cumsum(newly_infected - unobserved - confirmed, axis=1)))
    cum_susceptible = populations - cum_exposed - cum_unknown_infected - cum_confirmed - cum_deaths - cum_recovered - cum_unobserved
    log_lk = numpy.vstack((
        # Susceptible people get exposed to infected or tested
        binom.logpmf(newly_exposed,
                     cum_susceptible[:, :-1],
                     exposed_p * (
                         cum_unknown_infected[:, :-1] +
                         tested_contact_p[:, None] * cum_confirmed[:, :-1]) + (
                     external_sources_p)),
        # Exposed people become infected
        binom.logpmf(newly_infected,
                     cum_exposed[:, :-1],
                     infected_p),
        # People might recover or die before they are tested
        binom.logpmf(unobserved,
                     cum_unknown_infected[:, :-1],
                     dead_p[:, None] + immune_p),
        # Infected people become tested
        binom.logpmf(confirmed,
                     cum_unknown_infected[:, :-1] - unobserved,
                     tested_p[:, None]),
        # Tested people recover
        binom.logpmf(recovered,
                     cum_confirmed[:, :-1],
                     immune_p),
        # or they die
        binom.logpmf(deaths,
                     cum_confirmed[:, :-1],
                     dead_p[:, None]),
    ))
    errors = ~numpy.isfinite(log_lk)
    log_lk[errors] = 0
    prior = (
        beta.logpdf(dead_p, dead_alpha, dead_beta).sum(),
        beta.logpdf(tested_p, test_alpha, test_beta).sum(),
        beta.logpdf(tested_contact_p, contact_alpha, contact_beta).sum(),
    )
    return numpy.sum(log_lk) + sum(prior), errors

def propose(state):
    i = rand(30)
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
    elif i < 24:
        state["external_sources_p"] = 1e-10 * (numpy.random.random() - 0.5) + state["external_sources_p"]
    if i < 26:
        index = tuple(rand(state["newly_exposed"].shape))
        state["newly_exposed"][index] = 0
    elif i < 28:
        index = tuple(rand(state["newly_infected"].shape))
        state["newly_infected"][index] = 0
    elif i < 30:
        index = tuple(rand(state["unobserved"].shape))
        state["unobserved"][index] = 0

if __name__ == "__main__":
    try:
        state = json.load(open("state"))
        for key in state:
            if isinstance(state[key], list):
                state[key] = array(state[key])
        with open("samples.log", "r") as f:
            for line in f:
                s = line.split("\t")[0]
        s = int(s) + 1
        log = open("samples.log", "a")
    except (ValueError, FileNotFoundError, json.JSONDecodeError):
        state = dict(
            newly_exposed = numpy.zeros_like(confirmed),
            newly_infected = numpy.zeros_like(confirmed),
            unobserved = numpy.zeros_like(confirmed),
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
        state["newly_infected"][:, :-1] = confirmed[:, 1:]
        state["newly_exposed"][:, :-2] = confirmed[:, 2:]
        s = 0
        log = open("samples.log", "a")
        print(*
        ["s", "lk", "err", "exposed_p", "external_sources_p", "infected_p", "immune_p"] +
        ["tested_p {:}".format(c) for c in countries] +
        ["tested_contact_p {:}".format(c) for c in countries] +
        ["dead_p {:}".format(c) for c in countries],
        sep="\t",
        file=log)


    lk, err_a = log_lk(**state)
    err = numpy.inf if err_a is None else err_a.sum()
    while True:
        if s % 10000 == 0:
            print(*[s, lk, err, state["dead_alpha"], state["dead_beta"]])
            print(*[s, lk, err, state["exposed_p"], state["external_sources_p"], state["infected_p"], state["immune_p"]] +
                list(state["tested_p"]) +
                list(state["tested_contact_p"]) +
                list(state["dead_p"]),
                sep="\t",
                file=log)
            json.dump(state, open("state", "w"), default=numpy.ndarray.tolist, indent=2)
            log.flush()
        s += 1
        copy_state = deepcopy(state)
        propose(copy_state)
        new_lk, new_err_a = log_lk(**copy_state)
        new_err = numpy.inf if new_err_a is None else new_err_a.sum()
        if new_err < err or (new_err == err and new_lk > lk):
            state = copy_state
            lk = new_lk
            err = new_err
        elif new_err == err and numpy.random.random() < numpy.exp(new_lk - lk):
            state = copy_state
            lk = new_lk
            err = new_err

