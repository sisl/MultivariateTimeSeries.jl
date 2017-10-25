using MultivariateTimeSeries
using Base.Test

let
    srand(0)
    R = rand(20,3)
    d = DataFrame(R)
    mts = MTS(d, [1,6,11,16])
    labels = rand(1:3, 4)

    @test nrow(mts.data) == 20
    @test mts.index == [1,6,11,16]
    @test issubtype(eltype(mts.views), AbstractDataFrame)
    @test length(mts) == 4

    @test size(mts[1]) == (5,3)
    @test size(mts) == (4,)
    @test ncol(mts) == 3
    @test names(mts) == [:x1, :x2, :x3]
    @test length(collect(mts)) == length(mts.index)

    file = joinpath(dirname(@__FILE__), "mts.zip")
    write_data(file, mts, labels)

    @test has_labels(file)
    mts2, labels2 = read_data_labeled(file)
    mts3 = read_data(file)

    @test mts == mts2 == mts3
    @test labels == labels2
end

let
    srand(0)
    v = [DataFrame(rand(rand(3:5), 3)) for i=1:5]
    mts = MTS(v)
    @test all(x==y for (x,y) in zip(mts, v))
end
