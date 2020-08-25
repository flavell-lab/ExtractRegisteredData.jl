module ExtractRegisteredData

using LightGraphs, SimpleWeightedGraphs, DataStructures, SegmentationTools, ProgressMeter,
    Statistics, SparseArrays, LinearAlgebra, Arpack, Clustering, StatsBase, Plots,
    FlavellBase, MHDIO, ImageDataIO


include("extract_traces.jl")
include("run_transformix.jl")
include("register_neurons.jl")

export
    extract_traces_roi,
    run_transformix_centroids,
    run_transformix_roi,
    run_transformix_roi_graph,
    run_transformix_img,
    extract_traces_roi,
    extract_traces,
    error_rate,
    view_heatmap,
    register_neurons_nearest_neighbors,
    register_neurons_overlap,
    make_regmap_matrix,
    pairwise_dist,
    update_label_map,
    invert_label_map,
    find_neurons,
    delete_smeared_neurons
end # module
