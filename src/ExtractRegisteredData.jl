module ExtractRegisteredData

using Graphs, SimpleWeightedGraphs, DataStructures, SegmentationTools, ProgressMeter,
    Statistics, SparseArrays, LinearAlgebra, Arpack, Clustering, SparseClustering, StatsBase, Plots,
    FlavellBase, NRRDIO, ImageDataIO, PyPlot, SegmentationStats


include("extract_traces.jl")
include("run_transformix.jl")
include("register_neurons.jl")
include("merge.jl")
include("extract_roi_overlap.jl")

export
    run_transformix_centroids,
    run_transformix_roi,
    run_transformix_img,
    extract_traces,
    error_rate,
    register_neurons_overlap,
    make_regmap_matrix,
    make_regmap_matrix_,
    make_regmap_matrix_multicolor,
    pairwise_dist,
    update_label_map,
    invert_label_map,
    find_neurons,
    match_neurons_across_datasets,
    register_immobilized_rois,
    delete_smeared_neurons,
    extract_activity_am_reg,
    extract_roi_overlap,
    extract_roi_overlap_deepreg,
    extract_roi_overlap_deepreg_multicolor,
    output_roi_candidates,
    make_traces_array,
    merge_confocal_data!,
    get_centroids_preservenum,
    compute_centroid_dist_dict
end # module
