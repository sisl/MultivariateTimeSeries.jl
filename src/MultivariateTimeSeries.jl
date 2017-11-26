"""
    MultivariateTimeSeries

A package for storing and working with heterogeneous multivariate time series datasets.
"""
module MultivariateTimeSeries

using ZipFile
using Reexport
@reexport using DataFrames
using DataFrames.printtable

export
        MTS,
        has_labels,
        read_data,
        read_data_labeled,
        write_data

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
    views = Vector{SubDataFrame}(length(index))
    for i in eachindex(index)
        start_ind, end_ind = index[i], end_index(i, data, index) 
        views[i] = view(data, start_ind:end_ind)
    end
    MTS(data, index, views)
end
"""
    MTS(v::AbstractVector{DataFrame})

Create MTS object from a vector of DataFrames.  All dataframes must contain the same fields. 
"""
function MTS(v::AbstractVector{DataFrame})
    index_ = cumsum([nrow(d) for d in v])+1
    index = [1; index_[1:end-1]]
    data = vcat(v...)
    MTS(data, index)
end

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
    data = readtable(f_data)

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
    data = readtable(f_data)

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
    data = readtable(f_data)
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
    data = readtable(f_data)
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
    s = readstring(io)
    toks = split(s, ","; keep=false) #line1: indices
    index = parse.(Int, toks)
    index
end
"""
    read_labels(io::IO)

Read and parse dataset labels file.
"""
function read_labels(io::IO)
    typ = eval(parse(readline(io))) #line1: eltype
    labels = split(readline(io), ",") #line2: labels
    if typ in [Bool, Int, Int32, Int64, Float32, Float64] 
        labels = parse.(typ, labels)
    elseif typ in [Symbol]
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
    write_data{T}(zipfile::AbstractString, mts::MTS, labels::AbstractVector{T}=Int[])

Write data to file.  Optionally, include labels.
"""
function write_data{T}(zipfile::AbstractString, mts::MTS, labels::AbstractVector{T}=Int[])
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
    write_labels{T}(io::IO, labels::AbstractVector{T})

Write labels file.
"""
function write_labels{T}(io::IO, labels::AbstractVector{T})
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
    i = findfirst([f.name for f in r.files], file)
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
Base.start(mts::MTS) = 1 
Base.next(mts::MTS, i::Int) = (mts.views[i], i+1)
Base.done(mts::MTS, i::Int) = i > length(mts.views)
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
    DataFrames.ncol(mts::MTS) 

Returns the number of features in the dataset.
"""
DataFrames.ncol(mts::MTS) = ncol(mts.data)

end # module
