var documenterSearchIndex = {"docs":
[{"location":"neuronmatch/#Registration-based-neuron-matching-API","page":"Registration-based neuron matching API","title":"Registration-based neuron matching API","text":"","category":"section"},{"location":"neuronmatch/#Commonly-used-functions","page":"Registration-based neuron matching API","title":"Commonly used functions","text":"","category":"section"},{"location":"neuronmatch/","page":"Registration-based neuron matching API","title":"Registration-based neuron matching API","text":"extract_roi_overlap\nmake_regmap_matrix\nfind_neurons\nregister_immobilized_rois\ninvert_label_map","category":"page"},{"location":"neuronmatch/#ExtractRegisteredData.extract_roi_overlap","page":"Registration-based neuron matching API","title":"ExtractRegisteredData.extract_roi_overlap","text":"extract_roi_overlap(\n    best_reg::Dict, param_path::Dict, param::Dict; reg_dir_key::String=\"path_dir_reg\",\n    transformed_dir_key::String=\"path_dir_transformed\", reg_problems_key::String=\"path_reg_prob\",\n    param_path_moving::Union{Dict,Nothing}=nothing\n)\n\nExtracts ROI overlaps and activity differences.\n\nArguments\n\nbest_reg::Dict: Dictionary of best registration values.\nparam_path::Dict: Dictionary of paths to relevant files.\nparam::Dict: Dictionary of parameter values.\nreg_dir_key::String (optional): Key in param_path corresponding to the registration directory. Default path_dir_reg.\ntransformed_dir_key::String (optional): Key in param_path corresponding to the transformed ROI save directory. Default path_dir_transformed.\nreg_problems_key::String (optional): Key in param_path corresponding to the registration problems file. Default path_reg_prob.\nparam_path_moving::Union{Dict, Nothing}: If set, the param_path dictionary corresponding to the moving dataset. Otherwise, the method will assume\n\nthe moving and fixed datasets have the same path dictionary.\n\n\n\n\n\n","category":"function"},{"location":"neuronmatch/#ExtractRegisteredData.make_regmap_matrix","page":"Registration-based neuron matching API","title":"ExtractRegisteredData.make_regmap_matrix","text":"make_regmap_matrix(\n    roi_overlaps::Dict, roi_activity_diff::Dict, q_dict::Dict, best_reg::Dict, param::Dict;\n    watershed_errors::Union{Nothing,Dict}=nothing, max_fixed_t::Int=0\n)\n\nAssimilates ROIs from all time points and generates a quality-of-match matrix.\n\nArguments\n\nroi_overlaps::Dict: a dictionary of dictionaries of ROI matches for each pair of moving and fixed time points\nroi_activity_diff::Dict: a dictionaon of dictionaries of the difference between normalized red ROI acivity, for each pair of moving and fixed time points\nq_dict::Dict: a dictionary of dictionaries of dictionaries of the registration quality for each metric for each resolution for each pair of moving and fixed time points\nbest_reg::Dict: a dictionary of the registration that's being used, expressed as a tuple of resolutions, for each pair of moving and fixed time points\nparam::Dict: a dictionary of parameter settings to use, including:\nactivity_diff_threshold::Real: parameter that controls how much red-channel activity differences are penalized in the matrix.\n\nSmaller values = greater penalty. Default 0.3.     * watershed_error_penalty::Real: parameter that control how much watershed/UNet errors are penalized in the matrix. Smaller values = greater penalty. Default 0.5.     * quality_metric: metric to use in the q_dict parameter. Default NCC.     * matrix_self_weight::Real: amount of weight in the matrix to assign each ROI to itself. Default 0.5.     * size_mismatch_penalty::Real: penalty to apply to overlapping ROIs that don't fully overlap. Larger values = greater penalty. Default 2.\n\nwatershed_errors::Union{Nothing, Dict}: a dictionary of ROIs that might have had watershed or UNet errors for every time point.\n\nThis dictionary must be pre-shifted if moving and fixed datasets are not the same. Ignored if nothing.\n\nmax_fixed_t::Int: If the moving and fixed datasets are not the same, this number is added to each moving dataset time point   to distinguish the datasets in the matrix and label map.\n\nReturns\n\nregmap_matrix, a matrix whose (i,j)th entry encodes the quality of the match between ROIs i and j.\nlabel_map: a dictionary of dictionaries mapping original ROIs to new ROI labels, for each time point.\n\n\n\n\n\n","category":"function"},{"location":"neuronmatch/#ExtractRegisteredData.find_neurons","page":"Registration-based neuron matching API","title":"ExtractRegisteredData.find_neurons","text":"find_neurons(regmap_matrix, label_map::Dict, param::Dict)\n\nGroups ROIs into neurons based on a matrix of overlaps.\n\nArguments:\n\nregmap_matrix: Matrix of distances between the ROIs\nlabel_map::Dict: Dictionary of dictionaries mapping original ROIs to new ROI labels, for each time point.\nparam::Dict: Dictionary containing cluster_overlap_thresh and cluster_height_thresh parameter settings to use for clustering. \n\nReturns\n\nnew_label_map: Dictionary of dictionaries mapping original ROIs to neuron labels, for each time point.\ninv_map: Dictionary of dictionaries mapping time points to original ROIs, for each neuron label\nhmer: Raw clusters of the dataset.\n\n\n\n\n\n","category":"function"},{"location":"neuronmatch/#ExtractRegisteredData.register_immobilized_rois","page":"Registration-based neuron matching API","title":"ExtractRegisteredData.register_immobilized_rois","text":"register_immobilized_rois(regmap_matrix, label_map_regmap, inv_map_regmap, label_map_freelymoving, valid_rois_freelymoving, param, reg_timept)\n\nRegisters immobilized ROIs to the freely moving dataset.\n\nArguments:\n\nregmap_matrix: Matrix of distances between the ROIs\nlabel_map_regmap::Dict: Dictionary of dictionaries mapping original ROIs to new ROI labels for the freely-moving-to-immobilized registration.\ninv_map_regmap::Dict: Dictionary of dictionaries mapping new ROI labels to original ROIs, for the freely-moving-to-immobilized registration.\nlabel_map_freelymoving::Dict: Dictionary of dictionaries mapping original ROIs to new ROI labels in the freely-moving dataset (one of the outputs of find_neurons).\nvalid_rois_freelymoving: List of which ROIs correspond to neurons in the freely-moving dataset.\nparam::Dict: Dictionary containing max_t parameter setting to use for registration.\nreg_timept: Time point of the immobilized dataset to register.\n\nReturns:\n\nroi_matches: Dictionary of dictionaries mapping immobilized ROIs to matched freely moving ROIs.\ninv_matches: Dictionary of dictionaries mapping matched freely moving ROIs to immobilized ROIs.\nroi_matches_best: Dictionary of dictionaries mapping immobilized ROIs to matched freely moving ROIs, with only the best match for each immobilized ROI.\nroi_match_confidence: Dictionary of dictionaries mapping immobilized ROIs to the confidence of the match.\n\n\n\n\n\n","category":"function"},{"location":"neuronmatch/#ExtractRegisteredData.invert_label_map","page":"Registration-based neuron matching API","title":"ExtractRegisteredData.invert_label_map","text":"invert_label_map(label_map)\n\nInverts ROI label map label_map, so that new ROIs map back to dictionaries mapping original time points to original ROIs.\n\n\n\n\n\n","category":"function"},{"location":"neuronmatch/#transformix-helper-functions","page":"Registration-based neuron matching API","title":"transformix helper functions","text":"","category":"section"},{"location":"neuronmatch/","page":"Registration-based neuron matching API","title":"Registration-based neuron matching API","text":"run_transformix_centroids\nrun_transformix_roi\nrun_transformix_img","category":"page"},{"location":"neuronmatch/#ExtractRegisteredData.run_transformix_centroids","page":"Registration-based neuron matching API","title":"ExtractRegisteredData.run_transformix_centroids","text":"run_transformix_centroids(path, output, centroids, parameters, transformix_dir)\n\nRuns transformix to map centroids from the fixed volume to the moving volume. Note that this is the opposite of run_transformix_roi, because of the way transformix is implemented.\n\nArguments\n\npath::String: Path to directory containing elastix registration\noutput::String: Output file directory\ncentroids: File containing centroids to transform\nparameters::String: Transform parameter file\ntransformix_dir::String: Path to transformix executable\n\n\n\n\n\n","category":"function"},{"location":"neuronmatch/#ExtractRegisteredData.run_transformix_roi","page":"Registration-based neuron matching API","title":"ExtractRegisteredData.run_transformix_roi","text":"run_transformix_roi(path::String, input::String, output::String, parameters::String, parameters_roi::String, transformix_dir::String)\n\nRuns transformix to map an ROI image from the moving volume to the fixed volume.  Modifies various transform parameters to force nearest-neighbors interpolation. Returns the resulting transformed image. Note that this is the opposite of run_transformix_centroids, because of the way transformix is implemented.\n\nArguments\n\npath::String: Path to directory containing elastix registration. Other paths should be absolute; they are NOT relative to this.\ninput::String: Input file path\noutput::String: Output file directory path\nparameters::String: Path to transform parameter file\nparameters_roi::String: Filename to save modified parameters.\ntransformix_dir::String: Path to transformix executable\n\n\n\n\n\n","category":"function"},{"location":"neuronmatch/#ExtractRegisteredData.run_transformix_img","page":"Registration-based neuron matching API","title":"ExtractRegisteredData.run_transformix_img","text":"run_transformix_img(path::String, output::String, input::String, parameters::String, parameters_roi::String, transformix_dir::String; interpolation_degree::Int=1)\n\nRuns transformix to map an image from the moving volume to the fixed volume.  Returns the resulting transformed image. Note that this is the opposite of run_transformix_centroids, because of the way transformix is implemented.\n\nArguments\n\npath::String: Path to directory containing elastix registration\noutput::String: Output file directory\ninput::String: Input file path\nparameters::String: Transform parameter file\ntransformix_dir::String: Path to transformix executable\ninterpolation_degree::Int (optional): Interpolation degree. Default 1 (linear interpolation).\n\n\n\n\n\n","category":"function"},{"location":"neuronmatch/#Other-helper-functions","page":"Registration-based neuron matching API","title":"Other helper functions","text":"","category":"section"},{"location":"neuronmatch/","page":"Registration-based neuron matching API","title":"Registration-based neuron matching API","text":"pairwise_dist\nregister_neurons_overlap\nupdate_label_map\ndelete_smeared_neurons\nmerge_confocal_data!\nmatch_neurons_across_datasets","category":"page"},{"location":"neuronmatch/#ExtractRegisteredData.pairwise_dist","page":"Registration-based neuron matching API","title":"ExtractRegisteredData.pairwise_dist","text":"pairwise_dist(regmap_matrix; threshold::Real=1e-16, dtype::Type=Float64)\n\nFinds the distances between each pair of ROIs based on similarity between rows of the regmap_matrix.\n\nArguments\n\nregmap_matrix: matrix of registration matches\n\nOptional keyword arguments\n\nthreshold::Real: a small number added to the denominator to ensure there are no divide-by-zero errors. Default 1e-16.\ndtype::Type: Data type of matrix. Default Float64.\n\n\n\n\n\n","category":"function"},{"location":"neuronmatch/#ExtractRegisteredData.register_neurons_overlap","page":"Registration-based neuron matching API","title":"ExtractRegisteredData.register_neurons_overlap","text":"register_neurons_overlap(roi_inferred, roi, activity_inferred, activity)\n\nMatches ROIs from registered time points together based on the overlap heuristic.\n\nArguments\n\ninferred_roi: ROI image array from the moving volume that has had the registration applied to it (eg via transformix)\nroi: ROI image array from the fixed time point\ninferred_activity: array of activities of each ROI in the marker channel from the moving time point\nactivity: array of activities of each ROI in the marker channel from the fixed time point\n\nReturns\n\noverlapping_rois: a dictionary of ROI matches between the moving and fixed volumes.   The values of the dictionary are the amount of overlap as a fraction of total ROI size.   Eg: if ROI 15 in the moving volume has size 100, ROI 10 in the fixed volume has size 80, and the overlap has size 50,   (15, 10) => (0.5, 0.625) would be an entry in the dictionary.\nactivity_mismatch: a dictionary of the differences between the normalized activity of the ROIs   between the moving and fixed time points.\n\n\n\n\n\n","category":"function"},{"location":"neuronmatch/#ExtractRegisteredData.update_label_map","page":"Registration-based neuron matching API","title":"ExtractRegisteredData.update_label_map","text":"update_label_map(label_map, matches)\n\nUpdates ROI label map label_map to include ROI matches matches, and returns updated version.\n\n\n\n\n\n","category":"function"},{"location":"neuronmatch/#ExtractRegisteredData.delete_smeared_neurons","page":"Registration-based neuron matching API","title":"ExtractRegisteredData.delete_smeared_neurons","text":"delete_smeared_neurons(roi, threshold)\n\nDeletes neurons that might come from huge deformations in registration from an roi (neurons that have more pixels than threshold).\n\n\n\n\n\n","category":"function"},{"location":"neuronmatch/#ExtractRegisteredData.merge_confocal_data!","page":"Registration-based neuron matching API","title":"ExtractRegisteredData.merge_confocal_data!","text":"merge_confocal_data!(combined_data_dict::Dict, data_dict::Dict, data_dict_2::Dict, dataset::String; traces_key=\"traces_array\", zscored_traces_key=\"raw_zscored_traces_array\", F_F20_key=\"traces_array_F_F20\")\n\nThis function merges neuron identities from two datasets from the same animal, data_dict and data_dict_2, into a new dictionary combined_data_dict. The function is used to combine data from two different datasets.\n\nArguments\n\ncombined_data_dict::Dict: A dictionary that will contain the merged data.\ndata_dict::Dict: A dictionary containing the first dataset to be merged.\ndata_dict_2::Dict: A dictionary containing the second dataset to be merged.\ndataset::String: A string representing the name of the second dataset in the first dataset's dictionray.\ntraces_key::String: A string representing the key for the traces array in the dictionaries. Default is \"traces_array\".\nzscored_traces_key::String: A string representing the key for the zscored traces array in the dictionaries. Default is \"rawzscoredtraces_array\".\nF_F20_key::String: A string representing the key for the F/F0 traces array in the dictionaries. Default is \"tracesarrayF_F20\".\n\nThe function returns nothing, but modifies the combined_data_dict dictionary in place.\n\n\n\n\n\n","category":"function"},{"location":"neuronmatch/#ExtractRegisteredData.match_neurons_across_datasets","page":"Registration-based neuron matching API","title":"ExtractRegisteredData.match_neurons_across_datasets","text":"match_neurons_across_datasets(label_map_1::Dict, label_map_2::Dict, inv_map_reg::Dict, max_fixed_t::Int)\n\nMatches neurons across multiple datasets.\n\nArguments\n\nlabel_map_1::Dict: Label map of ROI/time points to neuron ID for dataset 1.\nlabel_map_2::Dict: Label map of ROI/time points to neuron ID for dataset 2.\ninv_map-reg::Dict: Map of neuron ID to ROI/time points for the registration between datasets. Time points in dataset 2 are shifted by max_fixed_t.\nmax_fixed_t::Int: Maximum time point in the first dataset.\n\n\n\n\n\n","category":"function"},{"location":"extract/#Traces-extraction-API","page":"Traces extraction API","title":"Traces extraction API","text":"","category":"section"},{"location":"extract/#Extract-activity-channel-traces","page":"Traces extraction API","title":"Extract activity channel traces","text":"","category":"section"},{"location":"extract/","page":"Traces extraction API","title":"Traces extraction API","text":"extract_activity_am_reg\nextract_traces\nmake_traces_array","category":"page"},{"location":"extract/#ExtractRegisteredData.extract_activity_am_reg","page":"Traces extraction API","title":"ExtractRegisteredData.extract_activity_am_reg","text":"extract_activity_am_reg(\n    param_path::Dict, param::Dict, shear_params_dict::Dict, crop_params_dict::Dict;\n    nrrd_path_key::String=\"path_dir_nrrd\", roi_dir_key::String=\"path_dir_roi_watershed\", \n    transform_key::String=\"name_transform_activity_marker_avg\"\n)\n\nExtracts activity marker activity from camera-alignment registration. Returns any errors encountered.\n\nArguments\n\nparam_path::Dict: Dictionary of paths to relevant files.\nparam::Dict: Directory of imaging parameters.\nshear_params_dict::Dict: Dictionary of shear-correction parameters used in the marker channel.\ncrop_params_dict::Dict: Dictionary of cropping parameters used in the marker channel.\nnrrd_path_key::String (optional): Key in param_path to path to unprocessed activity-channel NRRD files. Default path_dir_nrrd\nroi_dir_key::String (optional): Key in param_path to directory of neuron ROI files. Default path_dir_roi_watershed.\ntransform_key::String (optional): Key in param_path to file name of transform parameter files. Default path_dir_transformed_activity_marker_avg.\n\n\n\n\n\n","category":"function"},{"location":"extract/#ExtractRegisteredData.extract_traces","page":"Traces extraction API","title":"ExtractRegisteredData.extract_traces","text":"extract_traces(inverted_map, gcamp_data_dir)\n\nExtracts traces from neuron labels.\n\nArguments\n\ninverted_map: Dictionary of dictionaries mapping time points to original ROIs, for each neuron label\ngcamp_data_dir: Directory containing GCaMP activity data\n\n\n\n\n\n","category":"function"},{"location":"extract/#ExtractRegisteredData.make_traces_array","page":"Traces extraction API","title":"ExtractRegisteredData.make_traces_array","text":"make_traces_array(traces::Dict; threshold::Real=1, valid_rois=nothing, contrast::Real=1, replace_blank::Bool=false, cluster::Bool=false)\n\nTurns a traces dictionary into a traces array. Also outputs a heatmap of the array, and the labels of which neurons correspond to which rows.\n\nArguments\n\ntraces::Dict: Dictionary of traces.\nthreshold::Real (optional): Minimum number of time points necessary to include a neuron in the traces array.      Default 1 (all ROIs displayed). It is recommended to set either this or the valid_rois parameter.\nvalid_rois (optional): If set, use this set of neurons (in order) in the array. Set entries to nothing to insert gaps in the data      (eg: to preserve the previous labeling of neurons)\ncontrast::Real (optional): If set, increases the contrast of the heatmap. Does not change the output array.\nreplace_blank::Bool (optional): If set, replaces all blank entries in the traces (where the neuron was not found in that time point)      with the median activity for that neuron.\ncluster::Bool (optional): If set to true, cluster the neurons by activity similarity.\n\n\n\n\n\n","category":"function"},{"location":"extract/#SWF360-error-rate-computation","page":"Traces extraction API","title":"SWF360 error rate computation","text":"","category":"section"},{"location":"extract/","page":"Traces extraction API","title":"Traces extraction API","text":"error_rate","category":"page"},{"location":"extract/#ExtractRegisteredData.error_rate","page":"Traces extraction API","title":"ExtractRegisteredData.error_rate","text":"error_rate(traces; gfp_thresh::Real=1.5, num_thresh::Real=30)\n\nComputes the error rate of the traces, assuming they are all supposed to have constant activity.\n\nArguments\n\ntraces: the traces\n\nOptional keyword arguments\n\ngfp_thresh::Real: threshold amount for a neuron to be counted as having GFP (error rate is adjusted to account for mismatched neurons that have similar activity).   Assumption is that GFP-negative traces have value 0. If a trace ever differs from its median by more than this, it is counted as a mis-registration\nnum_thresh::Real: minimum amount of neurons needed to consider a trace\n\n\n\n\n\n","category":"function"},{"location":"extract/#Visualization-tools","page":"Traces extraction API","title":"Visualization tools","text":"","category":"section"},{"location":"extract/","page":"Traces extraction API","title":"Traces extraction API","text":"output_roi_candidates","category":"page"},{"location":"extract/#ExtractRegisteredData.output_roi_candidates","page":"Traces extraction API","title":"ExtractRegisteredData.output_roi_candidates","text":"output_roi_candidates(traces, inv_map::Dict, param_path::Dict, param::Dict, get_basename::Function, channel::Integer, t_range, valid_rois)\n\nOutputs neuron ROI candidates and a plot of their activity.\n\nArguments\n\ntraces: Array of traces for all ROIs\ninv_map::Dict: Dictionary that maps ROI identity to time points and UNet ROI labels\nparam_path::Dict: Dictionary of paths to relevant files\nparam::Dict: Dictionary of parameter values\nget_basename::Function: Function that gets the basename of an NRRD file\nchannel::Integer: Channel of the image to be displayed\nt_range: All time points\n\n\n\n\n\n","category":"function"},{"location":"#ExtractRegisteredData.jl-Documentation","page":"ExtractRegisteredData.jl Documentation","title":"ExtractRegisteredData.jl Documentation","text":"","category":"section"},{"location":"","page":"ExtractRegisteredData.jl Documentation","title":"ExtractRegisteredData.jl Documentation","text":"The ExtractRegisteredData.jl package matches neurons across time points using results from image segmentation and registration, and then extracts activity channel (GCaMP) traces of the detected neurons. It is designed to be used in the context of the ANTSUN pipeline.","category":"page"},{"location":"","page":"ExtractRegisteredData.jl Documentation","title":"ExtractRegisteredData.jl Documentation","text":"Pages = [\"neuronmatch.md\", \"extract.md\"]","category":"page"}]
}
