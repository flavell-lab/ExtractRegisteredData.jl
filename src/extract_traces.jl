"""
Extracts traces from transformed ROIs.

# Arguments

- `frames`: the set of frames to extract from
- `rootpath::String` is the working directory path. All other paths are relative to this.
- `roi_dir::String` contains ROIs transformed from a root frame into each other frame.
    The ROI file will be located at `rootpath/roi_dir/frame/result.mhd`
- `data_dir::String` contains the raw image data, for extracting traces.
    The file will be located at `rootpath/data_dir/img_prefix_tframe_chchannel.mhd`
- `img_prefix::String` is the image prefix
- `channel::Integer` is the channel

# Optional keyword arguments

- `enable_auto_cropping::Bool` (default false): if set to true, will automatically crop the images to the smaller of the two.
    Recommended to leave false to catch errors with giving the wrong input.
"""
function extract_traces_roi(frames, rootpath::String, roi_dir::String, mhd_dir::String, img_prefix::String, channel::Integer; enable_auto_cropping::Bool=false)
    data = Dict()
    n = length(frames)
    @showprogress for i=1:n
        frame = frames[i]
        img_roi = read_img(MHD(joinpath(rootpath, roi_dir, string(frame), "result.mhd")))
        img = read_mhd(rootpath, img_prefix, mhd_dir, frame, channel)
        if enable_auto_cropping
            ms = [min(size(img)[i], size(img_roi)[i]) for i=1:length(size(img))]
            img = img[1:ms[1], 1:ms[2], 1:ms[3]]
            img_roi = img_roi[1:ms[1], 1:ms[2], 1:ms[3]]
        end
        data[frame] = get_activity(img_roi, img)
    end
    return data
end


"""
Extracts traces from neuron labels.

# Arguments

- `inverted_map`: Dictionary of dictionaries mapping frames to original ROIs, for each neuron label
- `gcamp_data_dir`: Directory containing GCaMP activity data
"""
function extract_traces(inverted_map, gcamp_data_dir)
    traces = Dict()
    for roi in keys(inverted_map)
        traces[roi] = Dict()
        for frame in keys(inverted_map[roi])
            if length(inverted_map[roi][frame]) == 1
                activity = read_activity(joinpath(gcamp_data_dir, "$(frame).txt"))
                traces[roi][frame] = activity[inverted_map[roi][frame][1]]
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
    extracted_frames = Dict()
    err_frames = Dict()
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
        err_rois[roi] = [frame for frame in keys(trace) if (trace[frame] == trace[frame]) && (((trace[frame]/m) > factor) || ((m/trace[frame]) > factor))]
        for frame in keys(trace)
            if !(frame in keys(extracted_frames))
                extracted_frames[frame] = []
                err_frames[frame] = []
            end
            append!(extracted_frames[frame], roi)
            if frame in err_rois[roi]
                append!(err_frames[frame], roi)
            end
        end
        error += length(err_rois[roi])
        tot += length([v for v in vals])
    end
    gfp_frac = gfp/good_rois
    wrong_prob = 2 * gfp_frac * (1 - gfp_frac)
    return (err_rois, error/(tot*wrong_prob), gfp/good_rois, wrong_prob, extracted_frames, err_frames)
end

"""
Views heatmap for all `traces` that contain at least `threshold` frames.
"""
function view_heatmap(traces, threshold)
    valid_rois = [roi for roi in keys(traces) if length(keys(traces[roi])) >= threshold]
    frame_max = maximum([maximum(keys(traces[roi])) for roi in valid_rois])
    traces_arr = zeros((length(valid_rois), frame_max))
    count = 1
    for roi in valid_rois
        for frame in keys(traces[roi])
            traces_arr[count,frame] = traces[roi][frame]
        end
        count += 1
    end
    dist = pairwise_dist(traces_arr)
    cluster = hclust(dist, linkage=:single)
    ordered_traces_arr = zeros(size(traces_arr))
    for i=1:length(cluster.order)
        for j=1:frame_max
            ordered_traces_arr[i,j] = traces_arr[cluster.order[i],j]
        end
    end
    return heatmap(ordered_traces_arr)
end

