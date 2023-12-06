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
"""
function extract_roi_overlap_deepreg(problems::Vector, q_dict::Dict, param_path::Dict, param::Dict; reg_dir_key::String="path_dir_reg",
        transformed_dir_key::String="path_dir_transformed", fixed_roi_key::String="path_dir_roi_watershed_recropped",
        param_path_moving::Union{Dict,Nothing}=nothing)

    roi_overlaps = Vector{Dict}(undef, length(problems))
    roi_activity_diff = Vector{Dict}(undef, length(problems))
    errors = Vector{Exception}(undef, length(problems))

    if isnothing(param_path_moving)
        param_path_moving = param_path
    end

    @assert(length(params[dataset_central]["good_registration_resolutions"]) == 1)
    best = params[dataset_central]["good_registration_resolutions"][1]

    @showprogress for i in 1:length(problems)
        (moving, fixed) = problems[i]
        try
            dir = "$(moving)to$(fixed)"
            # Bspline registration failed
            if q_dict[(moving, fixed)][best]["NCC"] == Inf
                continue
            end

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