"""
    extract_roi_overlap(
        best_reg::Dict, param_path::Dict, param::Dict; reg_dir_key::String="path_dir_reg",
        transformed_dir_key::String="path_dir_transformed", reg_problems_key::String="path_reg_prob",
        param_path_moving::Union{Dict,Nothing}=nothing
    )

Extracts ROI overlaps and activity differences.

# Arguments
 - `best_reg::Dict`: Dictionary of best registration values.
 - `param_path::Dict`: Dictionary of paths to relevant files.
 - `param::Dict`: Dictionary of parameter values.
 - `reg_dir_key::String` (optional): Key in `param_path` corresponding to the registration directory. Default `path_dir_reg`.
 - `transformed_dir_key::String` (optional): Key in `param_path` corresponding to the transformed ROI save directory. Default `path_dir_transformed`.
 - `reg_problems_key::String` (optional): Key in `param_path` corresponding to the registration problems file. Default `path_reg_prob`.
 - `param_path_moving::Union{Dict, Nothing}`: If set, the `param_path` dictionary corresponding to the moving dataset. Otherwise, the method will assume
the moving and fixed datasets have the same path dictionary.
 - `alt_savepath::Union{Dict, Nothing}`: If set, save the modified ROI parameter files to this path instead of the original registration directory.
"""
function extract_roi_overlap(best_reg::Dict, param_path::Dict, param::Dict; reg_dir_key::String="path_dir_reg",
        transformed_dir_key::String="path_dir_transformed", reg_problems_key::String="path_reg_prob",
        param_path_moving::Union{Dict,Nothing}=nothing, alt_savepath::Union{String,Nothing}=nothing)

    problems = load_registration_problems([param_path[reg_problems_key]])

    roi_overlaps = Vector{Dict}(undef, length(problems))
    roi_activity_diff = Vector{Dict}(undef, length(problems))
    errors = Vector{Exception}(undef, length(problems))

    if isnothing(param_path_moving)
        param_path_moving = param_path
    end

    @showprogress for i in 1:length(problems)
        (moving, fixed) = problems[i]
        try
            dir = "$(moving)to$(fixed)"
            # Bspline registration failed
            if best_reg[(moving, fixed)][1] == 0
                continue
            end
            best = best_reg[(moving, fixed)]
            tf_base = joinpath(param_path[reg_dir_key], "$(dir)/TransformParameters.$(best[1]).R$(best[2])")
            if !isnothing(alt_savepath)
                create_dir(alt_savepath)
                create_dir(joinpath(alt_savepath, dir))
                tf_save = joinpath(alt_savepath, dir, "TransformParameters.$(best[1]).R$(best[2])")
            else
                tf_save = tf_base
            end
            img, result = run_transformix_roi(joinpath(param_path[reg_dir_key], "$(dir)"), 
                joinpath(param_path_moving["path_dir_roi_watershed"], "$(moving).nrrd"),  joinpath(param_path[transformed_dir_key], "$(dir)"), 
                "$(tf_base).txt", "$(tf_save)_roi.txt", param_path["path_transformix"])
            roi = read_img(NRRD(joinpath(param_path["path_dir_roi_watershed"], "$(fixed).nrrd")))
            roi_regmap = delete_smeared_neurons(img, param["smeared_neuron_threshold"])

            roi_overlap, roi_activity = register_neurons_overlap(roi_regmap, roi, 
                read_activity(joinpath(param_path_moving["path_dir_marker_signal"], "$(moving).txt")), 
                read_activity(joinpath(param_path["path_dir_marker_signal"], "$(fixed).txt")))
            roi_overlaps[i] = roi_overlap
            roi_activity_diff[i] = roi_activity
        catch e
            errors[i] = e
        end
    end

    roi_overlaps_dict = Dict()
    roi_activity_diff_dict = Dict()
    errors_dict = Dict()
    for (i, problem) in enumerate(problems)
        if isassigned(roi_overlaps, i)
            roi_overlaps_dict[problem] = roi_overlaps[i]
        end
        if isassigned(roi_activity_diff, i)
            roi_activity_diff_dict[problem] = roi_activity_diff[i]
        end
        if isassigned(errors, i)
            errors_dict[problem] = errors[i]
        end
    end
    if length(keys(errors_dict)) > 0
        @warn "Registration issues at some time points"
    end
    return roi_overlaps_dict, roi_activity_diff_dict, errors_dict
end


"""
`compute_centroid_dist_dict`

#### Description:
This function calculates the pairwise Euclidean distances between centroids in two different sets of region of interest (ROI) centroids (`roi_centroids_fixed` and `roi_centroids_transformed`). It filters and stores distances that are less than a specified maximum distance.

#### Parameters:
- **roi_centroids_fixed** (`Dict`): A dictionary where keys are identifiers for fixed images and values are matrices containing the centroid coordinates for each ROI in these images.
- **roi_centroids_transformed** (`Dict`): A dictionary where keys are tuples of identifiers for moving and fixed images, and values are matrices of centroid coordinates for each ROI in the moving images after transformation.
- **max_dist** (`Float`, optional): The maximum distance threshold for including a centroid pair in the output dictionary. The default value is 10.

#### Returns:
- **centroid_dist** (`Dict`): A nested dictionary where the primary key is a tuple containing identifiers for the moving and fixed images. The value is another dictionary where the key is a tuple of ROI indices (from moving and fixed images respectively), and the value is the calculated distance, provided it is less than `max_dist`.

#### Usage Example:
```julia
fixed_centroids = Dict("image1" => [1.0 2.0; 3.0 4.0])
transformed_centroids = Dict(("image2", "image1") => [1.1 2.1; 2.9 3.9])
distances = compute_centroid_dist_dict(fixed_centroids, transformed_centroids)
```
"""
function compute_centroid_dist_dict(roi_centroids_fixed, roi_centroids_transformed; max_dist=10)
    centroid_dist = Dict()
    @showprogress for (moving, fixed) in keys(roi_centroids_transformed)
        centroid_dist[(moving, fixed)] = Dict()
        for roi_moving in 1:size(roi_centroids_transformed[(moving,fixed)],1)
            for roi_fixed in 1:size(roi_centroids_fixed[fixed],1)
                if roi_centroids_fixed[fixed][roi_fixed,1] == -1 || roi_centroids_transformed[(moving, fixed)][roi_moving,1] == -1
                    continue
                end
                dist = euclidean_dist(roi_centroids_fixed[fixed][roi_fixed,:], roi_centroids_transformed[(moving, fixed)][roi_moving,:])
                if dist < max_dist
                    centroid_dist[(moving, fixed)][(roi_moving, roi_fixed)] = dist
                end
            end
        end
    end
    return centroid_dist
end

"""
`get_centroids_preservenum`

#### Description:
This function computes the centroids for each ROI in a given image. It supports specifying a maximum number of ROIs and allows for handling images where some ROIs may not be present.

#### Parameters:
- **img_roi** (`Array`): An array where each element's value corresponds to an ROI number in the image.
- **max_roi** (`Int`, optional): The maximum ROI number to consider. If not provided, the function calculates it as the maximum value in `img_roi`.
- **default_value** (`Int`, optional): The value to assign to centroids of ROIs that are not present in the image. The default is -1.

#### Returns:
- **centroids** (`Array`): A 2D array where each row corresponds to an ROI and contains the centroid coordinates of that ROI. Rows corresponding to absent ROIs are filled with the `default_value`.

#### Usage Example:
```julia
img = [1 1 0; 2 0 2]
centroids = get_centroids_preservenum(img)
```
"""
function get_centroids_preservenum(img_roi; max_roi=nothing, default_value=-1)
    if isnothing(max_roi)
        max_roi = maximum(img_roi)
    end

    centroids = zeros(max_roi, 3)
    for roi = 1:max_roi
        all_roi = findall(img_roi .== roi)
        if length(all_roi) == 0
            centroids[roi,:] .= -1
        else
            centroids[roi,:] .= centroid(all_roi)
        end
    end
    return centroids
end


"""
    extract_roi_overlap_deepreg
    (
        problems::Vector, q_dict::Dict, param_path::Dict, param::Dict; reg_dir_key::String="path_dir_reg",
        transformed_dir_key::String="path_dir_transformed", fixed_roi_key::String="path_dir_roi_watershed_recropped",
        param_path_moving::Union{Dict,Nothing}=nothing
    )

Extracts ROI overlaps and activity differences.

# Arguments
 - `problems::Vector`: Vector of registration problems.
 - `q_dict::Dict`: Dictionary of registration quality values.
 - `param_path::Dict`: Dictionary of paths to relevant files.
 - `param::Dict`: Dictionary of parameter values.
 - `reg_dir_key::String` (optional): Key in `param_path` corresponding to the registration directory. Default `path_dir_reg`.
 - `transformed_dir_key::String` (optional): Key in `param_path` corresponding to the transformed ROI save directory. Default `path_dir_transformed`.
 - `fixed_roi_key::String` (optional): Key in `param_path` corresponding to the fixed ROI save directory. Default `path_dir_roi_watershed_recropped`.
 - `param_path_moving::Union{Dict,Nothing}` (optional): Parameter path for moving data if it's different from fixed data.
"""
function extract_roi_overlap_deepreg(problems::Vector, param_path::Dict, param::Dict; reg_dir_key::String="path_dir_reg",
        transformed_dir_key::String="path_dir_transformed", fixed_roi_key::String="path_dir_roi_watershed_recropped",
        param_path_moving::Union{Dict,Nothing}=nothing)

    roi_overlaps = Vector{Dict}(undef, length(problems))
    roi_activity_diff = Vector{Dict}(undef, length(problems))
    errors = Vector{Exception}(undef, length(problems))

    if isnothing(param_path_moving)
        param_path_moving = param_path
    end

    @assert(length(param["good_registration_resolutions"]) == 1)
    best = param["good_registration_resolutions"][1]

    @showprogress for i in 1:length(problems)
        (moving, fixed) = problems[i]
        try
            dir = "$(moving)to$(fixed)"

            roi = read_img(NRRD(joinpath(param_path[fixed_roi_key], "$(fixed).nrrd")))
            img = read_img(NRRD(joinpath(param_path[transformed_dir_key], dir, "result.nrrd")))
            roi_regmap = delete_smeared_neurons(img, param["smeared_neuron_threshold"])

            roi_overlap, roi_activity = register_neurons_overlap(roi_regmap, roi, 
                read_activity(joinpath(param_path_moving["path_dir_marker_signal"], "$(moving).txt")), 
                read_activity(joinpath(param_path["path_dir_marker_signal"], "$(fixed).txt")))
            roi_overlaps[i] = roi_overlap
            roi_activity_diff[i] = roi_activity
        catch e
            errors[i] = e
        end
    end

    roi_overlaps_dict = Dict()
    roi_activity_diff_dict = Dict()
    errors_dict = Dict()
    for (i, problem) in enumerate(problems)
        if isassigned(roi_overlaps, i)
            roi_overlaps_dict[problem] = roi_overlaps[i]
        end
        if isassigned(roi_activity_diff, i)
            roi_activity_diff_dict[problem] = roi_activity_diff[i]
        end
        if isassigned(errors, i)
            errors_dict[problem] = errors[i]
        end
    end
    if length(keys(errors_dict)) > 0
        @warn "Registration issues at some time points"
    end
    return roi_overlaps_dict, roi_activity_diff_dict, errors_dict
end
