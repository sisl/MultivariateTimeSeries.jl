module MultivariateTimeSeries

using ZipFile
using Reexport
@reexport using DataFrames

export
    MTS,
    has_labels,
    read_data,
    read_data_labeled,
    write_data

const METAFILE = "_meta.txt"
const DATAFILE = "data.csv.gz"
const LABELFILE = "labels.txt"

struct MTS <: AbstractVector{SubDataFrame}
    data::DataFrame
    index::Vector{Int}
    views::Vector{SubDataFrame}
end
function MTS(data::DataFrame, index::Vector{Int})
    views = Vector{SubDataFrame}(length(index))
    for i in eachindex(index)
        start_ind, end_ind = index[i], end_index(i, data, index) 
        views[i] = view(data, start_ind:end_ind)
    end
    MTS(data, index, views)
end
function MTS(v::AbstractVector{DataFrame})
    index_ = cumsum([nrow(d) for d in v])+1
    index = [1; index_[1:end-1]]
    data = vcat(v...)
    MTS(data, index)
end

function end_index(i::Int, d::DataFrame, index::AbstractVector{Int})
    i+1 ≤ length(index) ? index[i+1]-1 : nrow(d)
end

function read_data(zipfile::AbstractString)
    r = ZipFile.Reader(zipfile)

    f_meta = open(r, METAFILE)
    index = read_meta(f_meta)

    f_data = open(r, DATAFILE)
    data = readtable(f_data)

    close(r)
    return MTS(data, index)
end
function read_data_labeled(zipfile::AbstractString)
    r = ZipFile.Reader(zipfile)

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
function read_meta(io::IO)
    s = readstring(io)
    toks = split(s, ","; keep=false) #line1: indices
    index = parse.(Int, toks)
    index
end
function read_labels(io::IO)
    typ = eval(parse(readline(io))) #line1: eltype
    labels = split(readline(io), ",") #line2: labels
    if typ != String
        labels = parse.(typ, labels)
    end
    labels
end

function has_labels(zipfile::AbstractString)
    r = ZipFile.Reader(zipfile)
    retval = LABELFILE in r
    close(r)
    retval
end

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
function write_meta(io::IO, mts::MTS)
    println(io, join(mts.index, ","))    
end
function write_labels{T}(io::IO, labels::AbstractVector{T})
    println(io, T)    
    println(io, join(labels, ","))    
end

function Base.in(file::AbstractString, r::ZipFile.Reader)
    file in [f.name for f in r.files]
end

function Base.open(r::ZipFile.Reader, file::AbstractString)
    i = findfirst([f.name for f in r.files], file)
    i > 0 || error("File not found: $file")
    return r.files[i] 
end

Base.getindex(mts::MTS, i::Int) = mts.views[i]
Base.length(mts::MTS) = length(mts.views)
Base.start(mts::MTS) = 1 
Base.next(mts::MTS, i::Int) = (mts.views[i], i+1)
Base.done(mts::MTS, i::Int) = i > length(mts.views)
function Base.:(==)(mts1::MTS, mts2::MTS)
    mts1.data == mts2.data
    mts1.index == mts2.index
    mts1.views == mts2.views
end
Base.names(mts::MTS) = names(mts.data)
Base.size(mts::MTS) = (length(mts),)

DataFrames.ncol(mts::MTS) = ncol(mts.data)

end # module
