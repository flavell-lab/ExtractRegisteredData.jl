"""
Matches ROIs from registered time points together based on the overlap heuristic.

# Arguments
- `inferred_roi`: ROI image array from the moving volume that has had the registration applied to it (eg via `transformix`)
- `roi`: ROI image array from the fixed time point
- `inferred_activity`: array of activities of each ROI in the marker channel from the moving time point
- `activity`: array of activities of each ROI in the marker channel from the fixed time point

# Returns
- `overlapping_rois`: a dictionary of ROI matches between the moving and fixed volumes.
    The values of the dictionary are the amount of overlap as a fraction of total ROI size.
    Eg: if ROI 15 in the moving volume has size 100, ROI 10 in the fixed volume has size 80, and the overlap has size 50,
    `(15, 10) => (0.5, 0.625)` would be an entry in the dictionary.

- `activity_mismatch`: a dictionary of the differences between the normalized activity of the ROIs
    between the moving and fixed time points.
"""
function register_neurons_overlap(inferred_roi, roi, inferred_activity, activity)
    rs = size(roi)
    is = size(inferred_roi)
    # the ROI images might be of different sizes, crop out excess
    ms = [min(rs[i], is[i]) for i=1:length(rs)]
    roi = roi[1:ms[1], 1:ms[2], 1:ms[3]]
    inferred_roi = inferred_roi[1:ms[1], 1:ms[2], 1:ms[3]]
    overlapping_rois = Dict()
    activity_mismatch = Dict()
    
    # normalize activities
    inferred_activity = inferred_activity ./ mean(inferred_activity)
    activity = activity ./ mean(activity)
    
    max_inf_roi = maximum(inferred_roi)
    max_roi = maximum(roi)
    
    points_inf_roi = []
    points_roi = []
    for ir in 1:max_inf_roi
        push!(points_inf_roi, get_points(inferred_roi, ir))
    end
    
    for r in 1:max_roi
        push!(points_roi, get_points(roi, r))
    end
    
    for ir in 1:max_inf_roi
        for r in 1:max_roi
            # look for ROI pairs that overlap significantly, possibly signifying a split in the matched ROI
            l = length([x for x in points_roi[r] if x in points_inf_roi[ir]])
            if l > 0
                l_roi = length(points_roi[r])
                l_inf = length(points_inf_roi[ir])
                overlapping_rois[(ir, r)] = (l / l_inf, l / l_roi)
                activity_mismatch[(ir, r)] = abs(activity[r] - inferred_activity[ir])
            end
        end
    end
    return (overlapping_rois, activity_mismatch)
end

"""
Given an ROI image array `img_roi`, returns a dictionary whose keys are centroids of ROIs, with values the corresponding ROI.
"""
function centroids_to_roi(img_roi)
    centroids = Dict()
    indices = CartesianIndices(img_roi)
    for i=1:maximum(img_roi)
        total = sum(img_roi .== i)
        if total == 0
            continue
        end
        centroids[map(x->round(x), Tuple(sum((img_roi .== i) .* indices))./total)] = i
    end
    return centroids
end

"""
Assimilates ROIs from all time points and generates a quality-of-match matrix.

# Arguments

- `roi_overlaps`: a dictionary of dictionaries of ROI matches for each pair of moving and fixed time points
- `roi_activity_diff`: a dictionaon of dictionaries of the difference between normalized red ROI acivity, for each pair of moving and fixed time points
- `q_dict`: a dictionary of dictionaries of dictionaries of the registration quality for each metric for each resolution for each pair of moving and fixed time points
- `best_reg`: a dictionary of the registration that's being used, expressed as a tuple of resolutions, for each pair of moving and fixed time points

# Optional keyword arguments

- `activity_diff_threshold::Real`: parameter that controls how much red-channel activity differences are penalized in the matrix.
Smaller values = greater penalty. Default 0.3.
- `watershed_error_penalty::Real`: parameter that control how much watershed/UNet errors are penalized in the matrix.
Smaller values = greater penalty. Default 0.5.
- `metric`: metric to use in the `q_dict` parameter. Default `NCC`.
- `self_weight::Real`: amount of weight in the matrix to assign each ROI to itself. Default 0.5.
- `size_mismatch_penalty::Real`: penalty to apply to overlapping ROIs that don't fully overlap.
    Larger values = greater penalty. Default 2.
- `watershed_errors`: a dictionary of ROIs that might have had watershed or UNet errors for every time point.

# Returns

- `regmap_matrix`, a matrix whose `(i,j)`th entry encodes the quality of the match between ROIs `i` and `j`.
- `label_map`: a dictionary of dictionaries mapping original ROIs to new ROI labels, for each time point.
"""
function make_regmap_matrix(roi_overlaps::Dict, roi_activity_diff::Dict, q_dict::Dict, best_reg::Dict;
        activity_diff_threshold::Real=0.3, watershed_error_penalty::Real=0.5, metric = "NCC", self_weight=0.5,
        size_mismatch_penalty=2, watershed_errors::Union{Nothing,Dict}=nothing)
    label_map = Dict()
    regmap_matrix = [Array{Int64,1}(), Array{Int64,1}(), Array{Float64,1}()]
    label_weight = Dict()
    count = 1
    for (moving, fixed) in keys(roi_overlaps)
        if !(moving in keys(label_map))
            label_map[moving] = Dict()
        end
        if !(fixed in keys(label_map))
            label_map[fixed] = Dict()
        end
        for (roi_moving, roi_fixed) in keys(roi_overlaps[(moving, fixed)])
            if !(roi_moving in keys(label_map[moving]))
                label_map[moving][roi_moving] = count
                count += 1
            end
            if !(roi_fixed in keys(label_map[fixed]))
                label_map[fixed][roi_fixed] = count
                count += 1
            end
            # weight of an edge is how well the ROIs overlap
            match_weight = minimum(roi_overlaps[(moving, fixed)][(roi_moving, roi_fixed)]) ^ size_mismatch_penalty
            # penalize for mNeptune activity being different
            match_weight *= activity_diff_threshold / (activity_diff_threshold + roi_activity_diff[(moving, fixed)][(roi_moving, roi_fixed)])
            # penalize for bad registration quality between time points
            match_weight /= (q_dict[(moving, fixed)][best_reg[(moving, fixed)]][metric])
            if watershed_errors != nothing && roi_moving in watershed_errors[moving]
                match_weight *= watershed_error_penalty
            end
            if watershed_errors != nothing && roi_fixed in watershed_errors[fixed]
                match_weight *= watershed_error_penalty
            end
            push!(regmap_matrix[1], label_map[moving][roi_moving])
            push!(regmap_matrix[2], label_map[fixed][roi_fixed])
            push!(regmap_matrix[3], match_weight)
            push!(regmap_matrix[2], label_map[moving][roi_moving])
            push!(regmap_matrix[1], label_map[fixed][roi_fixed])
            push!(regmap_matrix[3], match_weight)
        end
    end
    for i in 1:length(regmap_matrix[1])
        roi1 = regmap_matrix[1][i]
        label_weight[roi1] = get(label_weight, roi1, 0) + regmap_matrix[3][i]
    end
    m = mean(values(label_weight))
    if m == 0
        error("No successful registrations!")
    end
    for i in 1:length(regmap_matrix[1])
        roi1 = regmap_matrix[1][i]
        regmap_matrix[3][i] *= (1 - self_weight) / m
    end
    for i in 1:maximum(regmap_matrix[1])
        push!(regmap_matrix[1], i)
        push!(regmap_matrix[2], i)
        push!(regmap_matrix[3], self_weight)
    end

    regmap_matrix = sparse(regmap_matrix[1], regmap_matrix[2], regmap_matrix[3])
    return (regmap_matrix, label_map)
end

function make_regmap_matrix(roi_overlaps::Dict, roi_activity_diff::Dict, q_dict::Dict, best_reg::Dict, param::Dict; watershed_errors::Union{Nothing,Dict}=nothing)
    return make_regmap_matrix(roi_overlaps, roi_activity_diff, q_dict, best_reg,
        activity_diff_threshold=param["activity_diff_threshold"],
        watershed_error_penalty=param["watershed_error_penalty"],
        metric=param["quality_metric"],
        self_weight=param["matrix_self_weight"],
        size_mismatch_penalty=param["size_mismatch_penalty"],
        watershed_errors=watershed_errors)
end


"""
Finds the distances between each pair of ROIs based on similarity between rows of the `regmap_matrix`.

# Arguments

- `regmap_matrix`: matrix of registration matches

# Optional keyword arguments

- `threshold::Real`: a small number added to the denominator to ensure there are no divide-by-zero errors. Default 1e-16.
- `dtype::Type`: Data type of matrix. Default Float64.
"""
function pairwise_dist(regmap_matrix; threshold::Real=1e-16, dtype::Type=Float64)
    d = regmap_matrix * transpose(regmap_matrix)
    l = length(regmap_matrix[:,1])
    for i=1:l
        for j=i+1:l
            if j == i
                continue
            end
            d[i,j] = -d[i,j] / sqrt(d[i,i] * d[j,j] + threshold)
            d[j,i] = d[i,j]
        end
    end
    for i=1:l
        d[i,i] = -1
    end
    return map(x->Float32(x), d)
end

"""
Deletes neurons that might come from huge deformations in registration from an `roi` (neurons that have more pixels than `threshold`).
"""
function delete_smeared_neurons(roi, threshold)
    for i=1:maximum(roi)
        if sum(roi .== i) > threshold
            roi = collect(map(x->(x == i) ? 0 : x, roi))
        end
    end
    return roi
end


"""
Updates ROI label map `label_map` to include ROI matches `matches`, and returns updated version.
"""
function update_label_map(label_map, matches)
    new_label_map = Dict()
    for t in keys(label_map)
        new_label_map[t] = Dict()
        for roi in keys(label_map[t])
            new_label_map[t][roi] = matches[label_map[t][roi]]
        end
    end
    return new_label_map
end

"""
Inverts ROI label map `label_map`, so that new ROIs map back to dictionaries mapping original time points to original ROIs.
"""
function invert_label_map(label_map)
    inverted_map = Dict()
    for t in keys(label_map)
        for roi in keys(label_map[t])
            if !(label_map[t][roi] in keys(inverted_map))
                inverted_map[label_map[t][roi]] = Dict()
            end
            if t in keys(inverted_map[label_map[t][roi]])
                push!(inverted_map[label_map[t][roi]][t], roi)
            else
                inverted_map[label_map[t][roi]][t] = [roi]
            end
        end
    end
    return inverted_map
end



"""
Groups ROIs into neurons based on a matrix of overlaps.

# Arguments

- `regmap_matrix`: Matrix of distances between ROIs
- `label_map`: Dictionary of dictionaries mapping original ROIs to new ROI labels, for each time point.

# Optional keyword arguments

- `overlap_threshold::Real`: Maximum fraction of ROIs that can overlap in the same time point. Default 0.005
- `height_threshold::Real`: Maximum distance between ROIs from newly-added cluster. Default -0.01
- `dtype::Type`: Data type of distance matrix. Default Float32; can try setting to Float16 if out of memory.
- `pair_match::Bool`: Whether to only match pairs of frames (eg: in registering multiple datasets together)

# Returns

- `new_label_map`: Dictionary of dictionaries mapping original ROIs to neuron labels, for each time point.
- `inv_map`: Dictionary of dictionaries mapping time points to original ROIs, for each neuron label
- `hmer`: Raw clusters of the dataset.
"""
function find_neurons(regmap_matrix, label_map; overlap_threshold::Real=0.005, height_threshold::Real=-0.01, dtype::Type=Float64, pair_match::Bool=false)
    inv_map = invert_label_map(label_map)
    dist = pairwise_dist(regmap_matrix, dtype=dtype)
    hmer = hclust_minimum_threshold(dist, inv_map, overlap_threshold, pair_match=pair_match)
    n = length(keys(inv_map))
    c_to_roi = Dict()
    n_to_c = [-i for i=1:n]
    for i in 1:n
        c_to_roi[-i] = [i]
    end

    @showprogress for i = 1:length(hmer.heights)
        if hmer.heights[i] >= height_threshold
            break
        end
        merged = copy(c_to_roi[hmer.mleft[i]])
        append!(merged, c_to_roi[hmer.mright[i]])
        c_to_roi[i] = merged
        merged_rois = sort(map(x->collect(keys(inv_map[x]))[1], merged))
        for j in merged
            n_to_c[j] = i
        end
    end
    new_label_map = update_label_map(label_map, n_to_c)
    new_inv_map = invert_label_map(new_label_map);
    return (new_label_map, new_inv_map, hmer)
end

function find_neurons(regmap_matrix, label_map, param)
    return find_neurons(regmap_matrix, label_map,
        overlap_threshold=param["cluster_overlap_thresh"],
        height_threshold=param["cluster_height_thresh"])
end

