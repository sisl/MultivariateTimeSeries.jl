"""
    MultivariateTimeSeries

A package for storing and working with heterogeneous multivariate time series datasets.
"""
module MultivariateTimeSeries

using Random
using ZipFile
using Reexport
@reexport using DataFrames
using DataFrames: printtable
using CSV

export
        MTS,
        has_labels,
        read_data,
        read_data_labeled,
        write_data,
        normalize01,
        append_stop_token,
        test_mts,
        trim,
        transform,
        transform_cols

const METAFILE = "_meta.txt"
const DATAFILE = "data.csv.gz"
const LABELFILE = "labels.txt"

"""
    MTS

Multivariate time series data.
"""
struct MTS <: AbstractVector{SubDataFrame}
    data::DataFrame
    index::Vector{Int}
    views::Vector{SubDataFrame}
end

"""
    MTS(data::DataFrame, index::Vector{Int})

Create MTS object from a merged dataframe data and index containing the start indices of each record.
"""
function MTS(data::DataFrame, index::Vector{Int})
    views = Vector{SubDataFrame}(undef,length(index))
    for i in eachindex(index)
        start_ind, end_ind = index[i], end_index(i, data, index) 
        views[i] = view(data, start_ind:end_ind, :)
    end
    MTS(data, index, views)
end
"""
    MTS(v::AbstractVector{T}) where {T<:AbstractDataFrame}

Create MTS object from a vector of DataFrames.  All dataframes must contain the same fields. 
"""
function MTS(v::AbstractVector{T}) where {T<:AbstractDataFrame}
    index_ = cumsum([nrow(d) for d in v]) .+ 1
    index = [1; index_[1:end-1]]
    data = vcat(v...)
    MTS(data, index)
end
"""
    MTS(mts::MTS, inds::UnitRange{Int}) 

Create MTS object from mts using only data at inds.  
"""
MTS(mts::MTS, inds::UnitRange{Int}) = MTS(mts[inds]) 

"""
    end_index(i::Int, d::DataFrame, index::AbstractVector{Int})

Returns the end index of a record for index i.
"""
function end_index(i::Int, d::DataFrame, index::AbstractVector{Int})
    i+1 ≤ length(index) ? index[i+1]-1 : nrow(d)
end

"""
    Zip

Data is stored in a zip file
"""
struct Zip
    path::String
    Zip(f::String) = endswith(f, ".zip") ? new(f) : new("$f.zip")
end
"""
    Dir

Data is stored in a directory
"""
struct Dir
    path::String
    Dir(p::String) = new(abspath(p)) 
end

"""
    read_data(path::AbstractString)

Load and return a dataset.  Automatically detects the source type based on the given path.
"""
function read_data(path::AbstractString)
    if isdir(path)
        return read_data(Dir(path))
    else
        return read_data(Zip(path))
    end
end
"""
    read_data_labeled(path::AbstractString)

Load and return a labeled dataset as a tuple (data, labels)
"""
function read_data_labeled(path::AbstractString)
    if isdir(path)
        return read_data_labeled(Dir(path))
    else
        return read_data_labeled(Zip(path))
    end
end
"""
    read_data(z::Zip)

Load and return a dataset from a zip file.
"""
function read_data(z::Zip)
    r = ZipFile.Reader(z.path)

    f_meta = open(r, METAFILE)
    index = read_meta(f_meta)

    f_data = open(r, DATAFILE)
    data = DataFrame!(CSV.File(f_data))

    close(r)
    return MTS(data, index)
end
"""
    read_data_labeled(z::Zip)

Load and return a labeled dataset from a zip file.  Returns a tuple (data, labels)
"""
function read_data_labeled(z::Zip)
    r = ZipFile.Reader(z.path)

    f_meta = open(r, METAFILE)
    index = read_meta(f_meta)

    f_data = open(r, DATAFILE)
    data = DataFrame!(CSV.File(f_data))

    labels = nothing
    if LABELFILE in r 
        f_label = open(r, LABELFILE)
        labels = read_labels(f_label)
    end
    close(r)
    return (MTS(data, index), labels)
end
"""
    read_data(d::Dir)

Load and return a dataset from a directory.
"""
function read_data(d::Dir)
    f_meta = open(joinpath(d.path, METAFILE))
    index = read_meta(f_meta)
    close(f_meta)

    f_data = open(joinpath(d.path, DATAFILE))
    data = DataFrame!(CSV.File(f_data))
    close(f_data)

    return MTS(data, index)
end
"""
    read_data_labeled(d::Dir)

Load and return a labeled dataset from a directory.  Returns a tuple (data, labels).
"""
function read_data_labeled(d::Dir)
    f_meta = open(joinpath(d.path, METAFILE))
    index = read_meta(f_meta)
    close(f_meta)

    f_data = open(joinpath(d.path, DATAFILE))
    data = DataFrame!(CSV.File(f_data))
    close(f_data)

    labels = nothing
    if isfile(joinpath(d.path, LABELFILE)) 
        f_label = open(joinpath(d.path, LABELFILE))
        labels = read_labels(f_label)
        close(f_label)
    end
    return (MTS(data, index), labels)
end
"""
    read_meta(io::IO)

Read and parse dataset meta file.
"""
function read_meta(io::IO)
    s = read(io,String)
    toks = split(s, ","; keepempty=false) #line1: indices
    index = Base.parse.(Int, toks)
    index
end
"""
    read_labels(io::IO)

Read and parse dataset labels file.
"""
function read_labels(io::IO)
    typ = eval(Meta.parse(readline(io))) #line1: eltype
    labels = split(readline(io), ",") #line2: labels
    if typ in [Bool, Int, Int32, Int64, Float32, Float64] 
        labels = Base.parse.(typ, labels)
    elseif typ == Symbol
        labels = Symbol.(labels)
    else #fall back on convert
        labels = convert(Vector{typ}, labels)
    end
    labels
end

"""
    has_labels(path::AbstractString)

Returns true if dataset at path contains labels.
"""
function has_labels(path::AbstractString)
    if isdir(path)
        return has_labels(Dir(path))
    else
        return has_labels(Zip(path))
    end
end
"""
    has_labels(z::Zip)

Returns true if dataset in zipfile contains labels.
"""
function has_labels(z::Zip)
    r = ZipFile.Reader(z.path)
    retval = LABELFILE in r
    close(r)
    retval
end
"""
   has_labels(d::Dir) 

Returns true if dataset in directory contains labels.
"""
has_labels(d::Dir) = isfile(joinpath(d.path, LABELFILE))

"""
    write_data(zipfile::AbstractString, mts::MTS, labels::AbstractVector{T}=Int[]) where T

Write data to file.  Optionally, include labels.
"""
function write_data(zipfile::AbstractString, mts::MTS, labels::AbstractVector{T}=Int[]) where T
    w = ZipFile.Writer(zipfile)

    f_meta = ZipFile.addfile(w, METAFILE)
    write_meta(f_meta, mts)

    f_data = ZipFile.addfile(w, DATAFILE)
    printtable(f_data, mts.data)

    if !isempty(labels)
        f_labels = ZipFile.addfile(w, LABELFILE)
        write_labels(f_labels, labels)
    end

    close(w)
end
"""
    write_meta(io::IO, mts::MTS)

Write meta file.
"""
function write_meta(io::IO, mts::MTS)
    println(io, join(mts.index, ","))    
end
"""
    write_labels(io::IO, labels::AbstractVector{T}) where T

Write labels file.
"""
function write_labels(io::IO, labels::AbstractVector{T}) where T
    println(io, T)    
    println(io, join(labels, ","))    
end

"""
    Base.in(file::AbstractString, r::ZipFile.Reader)

Returns true if zipfile contains file.
"""
function Base.in(file::AbstractString, r::ZipFile.Reader)
    file in [f.name for f in r.files]
end

"""
    Base.open(r::ZipFile.Reader, file::AbstractString)

Returns file handle of file from zipfile.
"""
function Base.open(r::ZipFile.Reader, file::AbstractString)
    i = something(findfirst(isequal(file), [f.name for f in r.files]), 0)
    i > 0 || error("File not found: $file")
    return r.files[i] 
end

"""
    getindex(mts::MTS, i::Int) 

Returns the i'th record of the data.
"""
Base.getindex(mts::MTS, i::Int) = mts.views[i]

"""
    length(mts::MTS) 

Returns the number of records in the dataset.
"""
Base.length(mts::MTS) = length(mts.views)

#iterate over all records in dataset
function Base.iterate(mts::MTS)
    return (length(mts) > 0) ? (mts.views[1], 2) : nothing
end
function Base.iterate(mts::MTS, i::Int) 
    return (i > length(mts)) ? nothing : (mts.views[i], i+1)
end
function Base.:(==)(mts1::MTS, mts2::MTS)
    mts1.data == mts2.data
    mts1.index == mts2.index
    mts1.views == mts2.views
end

"""
    Base.names(mts::MTS)

Returns the feature names of the data.
"""
Base.names(mts::MTS) = names(mts.data)

"""
    Base.size(mts::MTS)

Returns the number of records in the dataset as a tuple
"""
Base.size(mts::MTS) = (length(mts),)

"""
    Base.minimum(mts::MTS, sym::Symbol)

Returns the minimum value over all records for feature sym.
"""
Base.minimum(mts::MTS, sym::Symbol) = minimum(mts.data[sym])
"""
    Base.minimum(mts::MTS) 

Returns the minimum values over all records for each feature. 
"""
Base.minimum(mts::MTS) = [minimum(mts.data[s]) for s in names(mts)]
"""
    Base.maximum(mts::MTS, sym::Symbol) 

Returns the maximum value over all records for feature sym.
"""
Base.maximum(mts::MTS, sym::Symbol) = maximum(mts.data[sym])
"""
    Base.maximum(mts::MTS) 

Returns the maximum values over all records for each feature.
"""
Base.maximum(mts::MTS) = [maximum(mts.data[s]) for s in names(mts)]
"""
    normalize01(mts::MTS)

Returns a new mts where each column is rescaled to [0,1].
"""
function normalize01(mts::MTS; fillval::Float64=0.5)
    mts1 = deepcopy(mts)
    mins, maxes = minimum(mts), maximum(mts)
    @assert length(mins) == length(maxes)
    for i = 1:length(mins) 
        d = maxes[i] - mins[i]
        mts1.data[i] = iszero(d) ? fill(fillval, nrow(mts1.data[i])) : (mts1.data[i] .- mins[i]) ./ d
    end
    mts1
end
"""
    DataFrames.ncol(mts::MTS) 

Returns the number of features in the dataset.
"""
DataFrames.ncol(mts::MTS) = ncol(mts.data)

function Base.vcat(m1::MTS, ms::MTS...) 
    offsets = vcat(nrow(m1.data), [nrow(m.data) for m in ms]...)[1:end-1] |> cumsum
    data = vcat(m1.data, [m.data for m in ms]...)
    index = vcat(m1.index, [m.index.+o for (m,o) in zip(ms,offsets)]...)
    MTS(data, index)
end

"""
    vec(mts::MTS) 

Convert to vec of vec of vec representation.  Some ML packages, e.g., Flux.jl, use this format.
"""
function Base.vec(mts::MTS)
    [vec(mts,i) for i=1:length(mts)]
end
function Base.vec(mts::MTS, i::Int) 
    A = convert(Array{Float64,2}, mts[i])
    [A[i,:] for i in 1:size(A,1)]
end

"""
    append_stop_token(mts::MTS)

Add a stop token column that is only true at the end of a sequence. Token type T.
"""
function append_stop_token(mts::MTS; T=Float64, colname=:stop)
    #update data
    V = []
    for i in eachindex(mts.index)
        start_ind, end_ind = mts.index[i], end_index(i, mts.data, mts.index)
        d = mts.data[start_ind:end_ind,:]
        d[colname] = zero(T)
        push!(d, fill(zero(T), ncol(d)))
        d[end, end] = one(T)
        push!(V, d)
    end
    data = vcat(V...)
    index = [mts.index[i]+(i-1) for i=1:length(mts.index)]
    MTS(data, index)
end
"""
    Base.repeat(mts::MTS, N::Int)

Stack mts vertically N times.
"""
Base.repeat(mts::MTS, N::Int) = foldl(vcat, mts for i=1:N)

"""
    Base.findfirst(f::Function, df::AbstractDataFrame)

Find the first row of the dataframe df where f is true.  Returns 0 if not found.
"""
function Base.findfirst(f::Function, df::AbstractDataFrame)
    i = 1
    for r in eachrow(df)
        f(r) && return i
        i += 1
    end
    return 0
end
"""
    Base.findfirst(f::Function, mts::MTS) 

Apply findfirst to each dataframe in mts returning a vector of integers that indicate the first row where findfirst is true. 
"""
Base.findfirst(f::Function, mts::MTS) = map(d->findfirst(f,d), mts)

"""
    transform(f::Function, mts::MTS)

Apply f to each dataframe in mts returning a new MTS object.
"""
function transform(f::Function, mts::MTS)
    dfs = Array{DataFrame}(undef,length(mts)) 
    for i in 1:length(mts)
        dfs[i] = copy(f(mts[i])) #defensive copy, needed?
    end
    MTS(dfs)
end

"""
    trim(mts::MTS, inds::Vector{Int})

Trim each dataframe in mts according to inds.  For each dataframe, rows 1:inds[i] are kept.  Invalid inds[i] are ignored.  A new MTS object is returned.
"""
function trim(mts::MTS, inds::Vector{Int})
    @assert length(mts) == length(inds)
    dfs = Array{DataFrame}(undef,length(inds)) 
    for i = 1:length(inds)
        rng_end = inds[i] in 1:nrow(mts[i]) ? inds[i] : nrow(mts[i])
        dfs[i] = convert(DataFrame, mts[i][1:rng_end,:]) 
    end
    MTS(dfs)
end

"""
    test_mts(rng::AbstractRNG=Random.G LOBAL_RNG)

Generate an MTS populated with some random values for testing use.
"""
function test_mts(rng::AbstractRNG=Random.GLOBAL_RNG)
	R = rand(rng, 20, 3)
    d = DataFrame(R)
    mts = MTS(d, [1, 6, 11, 16])
    mts
end

"""
    transform_cols(f::Function, mts::MTS)

Apply f to each column of all entries of mts
"""
function transform_cols(f::Function, mts::MTS)
    transform(mts) do d
        df = d[:,:]
        for c in names(df)
            df[c] = f(df[c])
        end
        df
    end
end

"""
    Base.zero(mts::MTS)

Apply zero function to all entries of mts
"""
Base.zero(mts::MTS) = transform_cols(zero, mts)

end # module
