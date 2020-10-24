"""
Matches ROIs from registered frames together, greedily taking the smallest-distance match between ROI centroids at each step.

# Arguments
- `inferred_roi`: ROI image array from the moving frame that has had the registration applied to it (eg via `transformix`)
- `roi`: ROI image array from the fixed frame
- `threshold`: Maximum distance allowable between two centroids

# Optional keyword arguments
- `xscale`, `yscale`, and `zscale`: Dimensions of each voxel, in um. Default 0.36um in x and y, and 1.0um in z.
"""
function register_neurons_nearest_neighbors(inferred_roi, roi, threshold; xscale::Real=0.36, yscale::Real=0.36, zscale::Real=1.0)
    inferred_centroids = centroids_to_roi(inferred_roi)
    centroids = centroids_to_roi(roi)
    graph = make_distance_graph(keys(inferred_centroids), keys(centroids); xscale=xscale, yscale=yscale, zscale=zscale)
    sorted_graph = sort(collect(graph), by=x->x[2])
    registered_infer = []
    registered = []
    # map from inferred centroids to centroids
    matches = Dict()
    match_weights = Dict()
    for m in sorted_graph
        match, dist = m
        match = (inferred_centroids[match[1]], centroids[match[2]])
        if dist > threshold
            return (matches, match_weights)
        end
        # one of the two nodes was already paired - collision
        if (match[1] in keys(matches)) || (match[2] in values(matches))
            continue
        end
        matches[match[1]] = match[2]
        match_weights[match[1]] = dist
    end
    return (matches, match_weights)
end

"""
Makes dictionary of distance values between two centroid point clouds.
Can optionally set `xscale`, `yscale`, and `zscale` to change voxel size in um.
"""
function make_distance_graph(centroids1, centroids2; xscale=0.36, yscale=0.36, zscale=1.0)
    graph = Dict()
    dist_func(c1,c2) = sqrt((xscale*(c1[1]-c2[1]))^2 + (yscale*(c1[2]-c2[2]))^2 + (zscale*(c1[3]-c2[3]))^2)
    for c1 in centroids1
        for c2 in centroids2
            graph[(c1,c2)] = dist_func(c1,c2)
        end
    end
    return graph
end


"""
Matches ROIs from registered frames together based on the overlap heuristic.

# Arguments
- `inferred_roi`: ROI image array from the moving frame that has had the registration applied to it (eg via `transformix`)
- `roi`: ROI image array from the fixed frame
- `inferred_activity`: array of activities of each ROI in the red channel from the moving frame
- `activity`: array of activities of each ROI in the red channel from the fixed frame

# Returns
- `overlapping_rois`: a dictionary of ROI matches between the moving and fixed frames.
    The values of the dictionary are the amount of overlap as a fraction of total ROI size.
    Eg: if ROI 15 in the moving frame has size 100, ROI 10 in the fixed frame has size 80, and the overlap has size 50,
    `(15, 10) => (0.5, 0.625)` would be an entry in the dictionary.

- `activity_mismatch`: a dictionary of the differences between the normalized activity of the ROIs
    in the moving and fixed frames.
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
Assimilates ROIs from all frames and generates a quality-of-match matrix.

# Arguments

- `roi_overlaps`: a dictionary of dictionaries of ROI matches for each pair of moving and fixed frames
- `roi_activity_diff`: a dictionaon of dictionaries of the difference between normalized red ROI acivity, for each pair of moving and fixed frames
- `q_dict`: a dictionary of dictionaries of dictionaries of the registration quality for each metric for each resolution for each pair of moving and fixed frames
- `best_reg`: a dictionary of the registration that's being used, expressed as a tuple of resolutions, for each pair of moving and fixed frames
- `watershed_errors`: a dictionary of ROIs that might have had watershed or UNet errors for every frame.

# Optional keyword arguments

- `activity_diff_threshold::Real`: parameter that controls how much red-channel activity differences are penalized in the matrix.
    Smaller values = greater penalty. Default 0.3.
- `watershed_error_penalty::Real`: parameter that control how much watershed/UNet errors are penalized in the matrix.
    Smaller values = greater penalty. Default 0.5.
- `metric`: metric to use in the `q_dict` parameter. Default `NCC`.
- `self_weight::Real`: amount of weight in the matrix to assign each ROI to itself. Default 0.5.
- `size_mismatch_penalty::Real`: penalty to apply to overlapping ROIs that don't fully overlap.
    Larger values = greater penalty. Default 2.

# Returns

- `regmap_matrix`, a matrix whose `(i,j)`th entry encodes the quality of the match between ROIs `i` and `j`.
- `label_map`: a dictionary of dictionaries mapping original ROIs to new ROI labels, for each frame.
"""
function make_regmap_matrix(roi_overlaps, roi_activity_diff, q_dict, best_reg, watershed_errors;
    activity_diff_threshold::Real=0.3, watershed_error_penalty::Real=0.5, metric = "NCC", self_weight=0.5, size_mismatch_penalty=2)
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
            # penalize for bad registration quality between frames
            match_weight /= (q_dict[(moving, fixed)][best_reg[(moving, fixed)]][metric])
            if roi_moving in watershed_errors[moving]
                match_weight *= watershed_error_penalty
            end
            if roi_fixed in watershed_errors[fixed]
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


"""
Finds the distances between each pair of ROIs based on similarity between rows of the `regmap_matrix`.

# Arguments

- `regmap_matrix`: matrix of registration matches

# Optional keyword arguments

- `threshold`: if distance is below this value, set it to 0.
- `dtype`: Data type of matrix. Default Float32; can possibly be set to Float16 if out of memory.
"""
function pairwise_dist(regmap_matrix; threshold=1e-8, dtype=Float32)
    d = regmap_matrix * transpose(regmap_matrix)
    l = length(regmap_matrix[:,1])
    for i=1:l
        for j=i+1:l
            if j == i
                continue
            end
            d[i,j] = -d[i,j] / sqrt(d[i,i] * d[j,j] + threshold ^ 2)
            d[j,i] = d[i,j]
        end
    end
    for i=1:l
        d[i,i] = -1
    end
    return map(x->Float32(x), d)
end

"""
Deletes excessively large neurons from an roi, such as those that might come from huge deformations in registration.
"""
function delete_smeared_neurons(roi; threshold=20000)
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
    for frame in keys(label_map)
        new_label_map[frame] = Dict()
        for roi in keys(label_map[frame])
            new_label_map[frame][roi] = matches[label_map[frame][roi]]
        end
    end
    return new_label_map
end

"""
Inverts ROI label map `label_map`, so that new ROIs map back to dictionaries mapping original frames to original ROIs.
"""
function invert_label_map(label_map)
    inverted_map = Dict()
    for frame in keys(label_map)
        for roi in keys(label_map[frame])
            if !(label_map[frame][roi] in keys(inverted_map))
                inverted_map[label_map[frame][roi]] = Dict()
            end
            if frame in keys(inverted_map[label_map[frame][roi]])
                push!(inverted_map[label_map[frame][roi]][frame], roi)
            else
                inverted_map[label_map[frame][roi]][frame] = [roi]
            end
        end
    end
    return inverted_map
end



"""
Groups ROIs into neurons based on a matrix of overlaps.

# Arguments

- `regmap_matrix`: Matrix of distances between ROIs
- `label_map`: Dictionary of dictionaries mapping original ROIs to new ROI labels, for each frame.

# Optional keyword arguments

- `overlap_threshold::Real`: Maximum fraction of ROIs that can overlap in the same frame. Default 0.05
- `height_threshold::Real`: Maximum distance between ROIs from newly-added cluster. Default -0.01
- `linkage`: Cluster linkage. Default `single` = merge clusters by closest points.

# Returns

- `new_label_map`: Dictionary of dictionaries mapping original ROIs to neuron labels, for each frame.
- `inv_map`: Dictionary of dictionaries mapping frames to original ROIs, for each neuron label
- `dtype::Type`: Data type of distance matrix. Default Float32; can try setting to Float16 if out of memory.
"""
function find_neurons(regmap_matrix, label_map; overlap_threshold::Real=0.05, height_threshold::Real=-0.01, linkage=:single, dtype::Type=Float32)
    inv_map = invert_label_map(label_map)
    dist = pairwise_dist(regmap_matrix, dtype=dtype)
    clusters = hclust(dist, linkage=linkage)
    c_to_roi = Dict()
    n_to_c = [-i for i=1:size(clusters.merges)[1]+1]
    for i in 1:size(clusters.merges)[1]+1
        c_to_roi[-i] = [i]
    end
    
    count = 0
    for i = 1:size(clusters.merges)[1]
        if clusters.heights[i] >= height_threshold
            break
        end
        merged = copy(c_to_roi[clusters.merges[i,1]])
        append!(merged, c_to_roi[clusters.merges[i,2]])
        c_to_roi[i] = merged
        merged_rois = sort(map(x->collect(keys(inv_map[x]))[1], merged))
        duplicates = sum([merged_rois[i] == merged_rois[i+1] for i=1:length(merged_rois)-1])
        if (duplicates / length(merged)) < overlap_threshold
            for j in merged
                n_to_c[j] = i
                count += 1
            end
        end
    end
    new_label_map = update_label_map(label_map, n_to_c)
    inv_map = invert_label_map(new_label_map)
    return (new_label_map, inv_map)
end

