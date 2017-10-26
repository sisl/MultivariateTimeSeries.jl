# MultivariateTimeSeries

A package for storing and working with heterogeneous multivariate time series data.

## Usage:

```julia
using MultivariateTimeSeries
using DataFrames

#create a dataset and labels from a vector of dataframes
v = [DataFrames(rand(5,3)) for i=1:10]
X, y,  = MTS(v), rand(Bool, 10)

#data records are dataframes and indexable
X[1]

#write data to file
write_data("mts.zip", X, y)

#read data from file
X2, y2 = read_data_labeled("mts.zip")
```

## Maintainers:

* Ritchie Lee, ritchie.lee@sv.cmu.edu

[![Build Status](https://travis-ci.org/sisl/MultivariateTimeSeries.jl.svg?branch=master)](https://travis-ci.org/sisl/MultivariateTimeSeries.jl) [![Coverage Status](https://coveralls.io/repos/sisl/MultivariateTimeSeries.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/sisl/MultivariateTimeSeries.jl?branch=master)
