"""
Extracts traces from neuron labels.

# Arguments

- `inverted_map`: Dictionary of dictionaries mapping time points to original ROIs, for each neuron label
- `gcamp_data_dir`: Directory containing GCaMP activity data
"""
function extract_traces(inverted_map, gcamp_data_dir)
    traces = Dict()
    for roi in keys(inverted_map)
        traces[roi] = Dict()
        for t in keys(inverted_map[roi])
            if length(inverted_map[roi][t]) == 1
                activity = read_activity(joinpath(gcamp_data_dir, "$(t).txt"))
                idx = inverted_map[roi][t][1]
                # Neuron out of FOV of green camera
                if idx > length(activity)
                    traces[roi][t] = 0
                else
                    traces[roi][t] = activity[idx]
                end
            end
        end
    end
    return traces
end

"""
Computes the error rate of the traces, assuming they are all supposed to have constant activity.

# Arguments

- `traces`: the traces

# Optional keyword arguments
- `factor::Real`: multiplicative amount by which activity must differ from median to count as an error. Default 1.5.
- `gfp_thresh::Real`: threshold amount for a neuron to be counted as having GFP (error rate is adjusted to account for mismatched neurons that have similar activity)
- `num_thresh::Real`: minimum amount of neurons needed to consider a trace
"""
function error_rate(traces; factor::Real=1.5, gfp_thresh::Real=170, num_thresh::Real=30)
    tot = 0
    error = 0
    gfp = 0
    good_rois = 0
    err_rois = Dict()
    extracted_timepts = Dict()
    err_timepts = Dict()
    for (roi, trace) in traces
        vals = [v for v in values(trace) if (v == v)]
        if length(trace) < num_thresh
            continue
        end
        good_rois += 1
        m = median(vals)
        if m > gfp_thresh
            gfp += 1
        end
        err_rois[roi] = [t for t in keys(trace) if (trace[t] == trace[t]) && (((trace[t]/m) > factor) || ((m/trace[t]) > factor))]
        for t in keys(trace)
            if !(t in keys(extracted_timepts))
                extracted_timepts[t] = []
                err_timepts[t] = []
            end
            append!(extracted_timepts[t], roi)
            if t in err_rois[roi]
                append!(err_timepts[t], roi)
            end
        end
        error += length(err_rois[roi])
        tot += length([v for v in vals])
    end
    gfp_frac = gfp/good_rois
    wrong_prob = 2 * gfp_frac * (1 - gfp_frac)
    return (err_rois, error/(tot*wrong_prob), gfp/good_rois, wrong_prob, extracted_timepts, err_timepts)
end

"""
Views heatmap for all `traces` that contain at least `threshold` time points.
"""
function view_heatmap(traces, threshold)
    valid_rois = [roi for roi in keys(traces) if length(keys(traces[roi])) >= threshold]
    t_max = maximum([maximum(keys(traces[roi])) for roi in valid_rois])
    traces_arr = zeros((length(valid_rois), t_max))
    count = 1
    for roi in valid_rois
        for t in keys(traces[roi])
            traces_arr[count,t] = traces[roi][t]
        end
        count += 1
    end
    dist = pairwise_dist(traces_arr)
    cluster = hclust(dist, linkage=:single)
    ordered_traces_arr = zeros(size(traces_arr))
    for i=1:length(cluster.order)
        for j=1:t_max
            ordered_traces_arr[i,j] = traces_arr[cluster.order[i],j]
        end
    end
    return heatmap(ordered_traces_arr)
end

"""
Turns a traces dictionary into a traces array.
Also outputs a heatmap of the array, and the labels of which neurons correspond to which rows.

# Arguments

 - `traces::Dict`: Dictionary of traces.
 - `threshold::Real` (optional): Minimum number of time points necessary to include a neuron in the traces array.
        Default 1 (all ROIs displayed). It is recommended to set either this or the `valid_rois` parameter.
 - `valid_rois` (optional): If set, use this set of neurons (in order) in the array.
 - `normalized::Bool` (optional): If set to true, normalize each neuron's trace (by dividing by its median).
 - `contrast::Real` (optional): If set, increases the contrast of the heatmap. Does not change the output array.
 - `replace_blank::Bool` (optional): If set, replaces all blank entries in the traces (where the neuron was not found in that time point)
        with the median activity for that neuron.
"""
function make_traces_array(traces::Dict; threshold::Real=1, valid_rois=nothing, normalized::Bool=false, contrast::Real=1, replace_blank::Bool=false)
    if valid_rois == nothing
        valid_rois = [roi for roi in keys(traces) if length(keys(traces[roi])) >= threshold]
    end
    t_max = maximum([maximum(keys(traces[roi])) for roi in valid_rois])
    traces_arr = zeros((length(valid_rois), t_max))
    count = 1
    for roi in valid_rois
        med = median([x for x in values(traces[roi]) if x == x])
        for t = 1:t_max
            if t in keys(traces[roi])
                traces_arr[count,t] = traces[roi][t]
                if normalized
                    traces_arr[count,t] = traces_arr[count,t] ./ med
                end
            elseif replace_blank
                traces_arr[count,t] = 1 + !normalized * (med - 1)
            else
                traces_arr[count,t] = 0
            end
        end
        count += 1
    end
    max_val = maximum(traces_arr)
    new_traces_arr = map(x->min(x,max_val/contrast), traces_arr)
    return (traces_arr, heatmap(new_traces_arr), valid_rois)
end


"""
Extracts activity marker activity from camera-alignment registration. Returns any errors encountered.

# Arguments
 - `param_path::Dict`: Dictionary of paths to relevant files.
 - `mhd_path::String`: Directory of MHD files to extract activity from.
 - `get_basename::Function`: Function that returns basename of mhd file given time point and channel
 - `t_range`: Range of time points to extract over. Make sure to exclude ts where the registration failed.
 - `ch_activity`: Channel that corresponds to the activity marker.
"""
function extract_activity_am_reg(param_path::Dict, mhd_path::String, get_basename::Function, t_range, ch_activity)
    errors = Dict()
    @showprogress for t in t_range
        try
            mhd_str = joinpath(param_path["path_root_process"], mhd_path, get_basename(t, ch_activity)*".mhd")
            img = read_img(MHD(mhd_str))
            regpath = joinpath(param_path["path_dir_reg_activity_marker"], "$(t)to$(t)")
            create_dir(joinpath(param_path["path_dir_transformed_activity_marker"], "$(t)"))
            run_transformix_roi(param_path["path_dir_reg_activity_marker"], joinpath(param_path["path_dir_roi_watershed"], "$(t).mhd"),
                joinpath(param_path["path_dir_transformed_activity_marker"], "$(t)"), joinpath(regpath, param_path["name_transform_activity_marker"]),
                joinpath(regpath, param_path["name_transform_activity_marker_roi"]), param_path["path_transformix"])
            img_roi = read_img(MHD(joinpath(param["path_dir_transformed_activity_marker"], "$(t)", "result.mhd")))
            
            # get activity
            activity = get_activity(img_roi, img)
            activity_file = joinpath(rootpath, param_path["path_dir_am_activity"], "$(t).txt") 
            write_activity(activity, activity_file)
        catch e
            errors[t] = e
        end
    end
    if length(keys(errors)) > 0
        @warn "Unsuccessful camera alignment registration at some time points"
    end
    return errors
end

"""
Extracts ROI overlaps and activity differences.

# Arguments
 - `param_path::Dict`: Dictionary of paths to relevant files.
 - `param::Dict`: Dictionary of parameter values.
"""
function extract_roi_overlap(param_path::Dict, param::Dict)
    roi_overlaps = Dict()
    roi_activity_diff = Dict()
    errors = Dict()

    @showprogress for (moving, fixed) in problems
        try
            dir = "$(moving)to$(fixed)"
            # Bspline registration failed
            if best_reg[(moving, fixed)][1] == 0
                continue
            end
            best = best_reg[(moving, fixed)]
            tf_dir = 
            tf_base = joinpath(param_path["path_dir_reg"], "$(dir)/TransformParameters.$(best[1]).R$(best[2])")
            img, result = run_transformix_roi(joinpath(param_path["path_dir_reg"], "$(dir)"), 
                joinpath(param_path["path_dir_roi_watershed"], "$(moving).mhd"),  joinpath(param_path["path_dir_transformed"], "$(dir)"), 
                joinpath(rootpath, "$(tf_base).txt"), joinpath(rootpath, "$(tf_base)_roi.txt"), param_path["path_transformix"])
            roi = read_img(MHD(joinpath(param_path["path_dir_roi_watershed"], "$(fixed).mhd")))
            roi_regmap = delete_smeared_neurons(img, param["smeared_neuron_threshold"])

            roi_overlap, roi_activity = register_neurons_overlap(roi_regmap, roi, 
                read_activity(joinpath(param_path["path_dir_activity"], "$(moving).txt")), 
                read_activity(joinpath(param_path["path_dir_activity"], "$(fixed).txt")))
            roi_overlaps[(moving, fixed)] = roi_overlap
            roi_activity_diff[(moving, fixed)] = roi_activity
        catch e
            errors[(moving, fixed)] = e
        end
    end

    if length(keys(errors) > 0)
        @warn "Registration issues at some time points"
    end
    return roi_overlaps, roi_activity_diff, errors
end


"""
Outputs neuron ROI candidates and a plot of their activity.

# Arguments
 - `traces::Dict`: Dictionary of traces for all ROIs
 - `inv_map::Dict`: Dictionary that maps ROI identity to time points and UNet ROI labels
 - `param_path::Dict`: Dictionary of paths to relevant files
 - `param::Dict`: Dictionary of parameter values
 - `get_basename::Function`: Function that gets the basename of an MHD file
 - `channel::Integer`: Channel of the image to be displayed
 - `t_range`: All time points
"""
function output_roi_candidates(traces::Dict, inv_map::Dict, param_path::Dict, param::Dict, get_basename::Function, channel::Integer, t_range)
    @showprogress for neuron in [x for x in keys(traces) if length(keys(traces[x])) >= param["num_detections_threshold"]]
        min_t = minimum(keys(inv_map[neuron]))
        img = maxprj(Float64.(read_img(MHD(get_basename(min_t, channel)*".mhd"))), dims=3);
        
        centroids = read_centroids_roi(joinpath(param_path["path_dir_centroid"], "$(min_t).txt"))
        roi = centroids[inv_map[neuron][min_t][1]][1:2]
        
        fig, axes = subplots(ncols=2, figsize=(12,6));
        sorted_times = sort(collect(keys(traces[neuron])));

        neuron_traces = [traces[neuron][t] for t in sorted_times]
        axes[1].imshow(img);
        axes[1].scatter([roi[2]], [roi[1]], c="r", s=4);
        axes[2].scatter(sorted_times, neuron_traces);
        axes[2].set_ylim(0, maximum(neuron_traces));
        axes[2].set_xlim(minimum(t_range), maximum(t_range));
        PyPlot.savefig(joinpath(param_path["path_roi_candidates"], string(neuron)*".png"));
     end
end
