module ExtractRegisteredData

using LightGraphs, SimpleWeightedGraphs, DataStructures, SegmentationTools, ProgressMeter,
    Statistics, SparseArrays, LinearAlgebra, Arpack, Clustering, StatsBase, Plots,
    FlavellBase, MHDIO, ImageDataIO, PyPlot


include("extract_traces.jl")
include("run_transformix.jl")
include("register_neurons.jl")

export
    run_transformix_centroids,
    run_transformix_roi,
    run_transformix_img,
    extract_traces,
    error_rate,
    view_heatmap,
    register_neurons_overlap,
    make_regmap_matrix,
    pairwise_dist,
    update_label_map,
    invert_label_map,
    find_neurons,
    delete_smeared_neurons,
    extract_activity_am_reg,
    extract_roi_overlap,
    output_roi_candidates,
    make_traces_array
end # module
