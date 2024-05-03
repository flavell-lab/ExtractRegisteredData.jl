"""
    register_neurons_overlap(roi_inferred, roi, activity_inferred, activity)

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
function register_neurons_overlap(roi_inferred, roi, activity_inferred, activity)
    size_roi = size(roi)
    size_roi_inf = size(roi_inferred)
    
    size_crop = [min(size_roi[i], size_roi_inf[i]) for i=1:length(size_roi)]
    
    roi = roi[1:size_crop[1], 1:size_crop[2], 1:size_crop[3]]
    roi_inferred = roi_inferred[1:size_crop[1], 1:size_crop[2], 1:size_crop[3]]

    overlapping_rois = Dict()
    activity_mismatch = Dict()

    # normalize activities
    activity_inferred = activity_inferred ./ mean(activity_inferred)
    activity = activity ./ mean(activity)

    max_roi_inf = maximum(roi_inferred)
    max_roi = maximum(roi)

    idx_nz_roi = findall(roi .!= 0)
    idx_nz_roi_inferred = findall(roi_inferred .!= 0)

    count_overlap = zeros(max_roi_inf, max_roi)

    n_roi = zeros(max_roi)
    n_roi_inferred = zeros(max_roi_inf)
    for i = idx_nz_roi
        roi_id = roi[i]
        n_roi[roi_id] += 1
    end

    for i = idx_nz_roi_inferred
        roi_inferred_id = roi_inferred[i]
        if roi_inferred_id > 0
            n_roi_inferred[roi_inferred_id] += 1
            roi_id = roi[i]
            if roi_id > 0
                count_overlap[roi_inferred_id,roi_id] += 1
            end
        end
    end

    for i = 1:max_roi_inf
        nz = count_overlap[i,:]
        for j = findall(nz .> 0)
            overlapping_rois[(i,j)] = (nz[j] / n_roi_inferred[i], nz[j] / n_roi[j])
            activity_mismatch[(i,j)] = abs(activity[j] - activity_inferred[i])
        end
    end
    
    return (overlapping_rois, activity_mismatch)
end

"""
    centroids_to_roi(img_roi)

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

function make_regmap_matrix(centroid_dist_dict::Dict, roi_overlaps::Dict, q_dict::Dict, best_reg::Dict, regularization_dict::Dict, displacement_dict::Dict, param_path::Dict;
        overlap_weight::Real=2.0, centroid_weight::Real=1.0, activity_diff_weight::Real=3.0, q_weight=25.0, regularization_weight=4.0, displacement_weight=2.0, self_weight=1.0, 
        metric = "NCC", regularization_key="nonrigid_penalty", max_fixed_t::Int=0, zero_overlap_val=1e-10, max_dist=10, min_weight=1e-6
    )

    label_map = Dict()
    regmap_matrix = [Array{Int64,1}(), Array{Int64,1}(), Array{Float64,1}()]
    label_weight = Dict()
    count = 1

    for (moving, fixed) in keys(centroid_dist_dict)
        moving_shifted = moving + max_fixed_t
        if !(moving_shifted in keys(label_map))
            label_map[moving_shifted] = Dict()
        end
        if !(fixed in keys(label_map))
            label_map[fixed] = Dict()
        end
        roi_pairs = collect(keys(centroid_dist_dict[(moving, fixed)]))
        append!(roi_pairs, collect(keys(roi_overlaps[(moving, fixed)])))
        roi_pairs = unique(roi_pairs)

        activity_fixed = read_activity(joinpath(param_path["path_dir_marker_signal"], "$(fixed).txt"))
        activity_fixed = activity_fixed ./ mean(activity_fixed)
        activity_moving = read_activity(joinpath(param_path["path_dir_marker_signal"], "$(moving).txt"))
        activity_moving = activity_moving ./ mean(activity_moving)
        
        for (roi_moving, roi_fixed) in roi_pairs
            # weight of an edge is how well the ROIs overlap
            match_weight = minimum(get(roi_overlaps[(moving, fixed)], (roi_moving, roi_fixed), (zero_overlap_val, zero_overlap_val))) ^ overlap_weight
            # penalize for centroids being far apart
            match_weight *= exp(-centroid_weight * get(centroid_dist_dict[(moving, fixed)], (roi_moving, roi_fixed), max_dist))
            # penalize for mNeptune activity being different
            activity_mismatch = abs(activity_moving[roi_moving] - activity_fixed[roi_fixed])
            match_weight *= exp(-activity_diff_weight * activity_mismatch)
            # penalize for bad registration quality between time points
            match_weight *= (q_dict[(moving, fixed)][best_reg[(moving, fixed)]][metric]) ^ q_weight
            # penalize for large regularization weight
            match_weight *= exp(-regularization_weight * regularization_dict[(moving, fixed)][regularization_key])
            # penalize for individual ROIs moving too much
            match_weight /= (1.0 + displacement_weight * displacement_dict[(moving, fixed)][roi_fixed])

            if match_weight < min_weight
                continue
            end

            if !(roi_moving in keys(label_map[moving_shifted]))
                label_map[moving_shifted][roi_moving] = count
                count += 1
            end
            if !(roi_fixed in keys(label_map[fixed]))
                label_map[fixed][roi_fixed] = count
                count += 1
            end

            push!(regmap_matrix[1], label_map[moving_shifted][roi_moving])
            push!(regmap_matrix[2], label_map[fixed][roi_fixed])
            push!(regmap_matrix[3], match_weight)
            push!(regmap_matrix[2], label_map[moving_shifted][roi_moving])
            push!(regmap_matrix[1], label_map[fixed][roi_fixed])
            push!(regmap_matrix[3], match_weight)
        end
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
`make_regmap_matrix`

#### Description:
Generates a matrix that encapsulates the quality of match between regions of interest (ROIs) across different time points or imaging conditions, considering various factors such as overlap, centroid proximity, activity difference, registration quality, and more. This function is typically used to aid in image registration or fusion tasks by providing a robust way to assess how well different regions align or correspond across datasets.

#### Parameters:
- **centroid_dist_dict** (`Dict`): A dictionary containing the distances between centroids of corresponding ROIs in different images.
- **roi_overlaps** (`Dict`): Dictionary detailing the overlap metrics between ROIs across the compared images.
- **q_dict** (`Dict`): Dictionary that contains the registration quality metrics for each pair of compared images.
- **best_reg** (`Dict`): Specifies the best registration configurations used between image pairs.
- **regularization_dict** (`Dict`): Contains regularization penalties applied during the registration process.
- **displacement_dict** (`Dict`): Contains displacement values for ROIs, indicating how much an ROI has moved from one image to another.
- **param_path** (`Dict`): Dictionary specifying paths to necessary parameter files or data.
- **overlap_weight** (`Real`, optional): Weight given to the overlap metric in the overall match score (default: 2.0).
- **centroid_weight** (`Real`, optional): Weight applied to the centroid distance in the match score, influencing how centroid proximity affects the scoring (default: 1.0).
- **activity_diff_weight** (`Real`, optional): Weight for the difference in activity (e.g., fluorescence intensity) between matched ROIs, influencing the score based on functional similarity (default: 3.0).
- **q_weight** (`Real`, optional): Exponential weight applied to the quality of registration metric from `q_dict` (default: 25.0).
- **regularization_weight** (`Real`, optional): Weight for the regularization penalty affecting the match score (default: 4.0).
- **displacement_weight** (`Real`, optional): Weight given to the displacement of ROIs, affecting the match score based on how much ROIs have moved (default: 2.0).
- **self_weight** (`Real`, optional): Weight for self-matching of ROIs, ensuring a baseline self-affinity (default: 1.0).
- **metric** (`String`, optional): Specifies the metric used to assess registration quality from `q_dict` (default: "NCC").
- **regularization_key** (`String`, optional): Key to access the specific regularization metric from `regularization_dict` (default: "nonrigid_penalty").
- **max_fixed_t** (`Int`, optional): Maximum time offset for fixed datasets, used to adjust labels or indices in time-shifted analyses (default: 0).
- **zero_overlap_val** (`Float`, optional): Small value used to avoid division by zero or log of zero in calculations involving no overlap (default: 1e-10).
- **max_dist** (`Int`, optional): Maximum allowable distance for considering centroid proximity (default: 10).
- **min_weight** (`Float`, optional): Minimum threshold for the weight of a match to be included in the final matrix (default: 1e-6).

#### Returns:
- **Tuple**: Contains the `regmap_matrix`, a sparse matrix where each element (i, j) represents the computed match score between ROI i and j, and `label_map`, a dictionary mapping each original ROI to a new numerical label for easier reference in matrix operations.

#### Usage Example:
```julia
# Define parameters and data structures
centroid_dist = Dict(...)
roi_overlaps = Dict(...)
q_dict = Dict(...)
best_reg = Dict(...)
regularization_dict = Dict(...)
displacement_dict = Dict(...)
param_path = Dict(...)

# Generate registration map matrix
(regmap_matrix, label_map) = make_regmap_matrix(centroid_dist, roi_overlaps, q_dict, best_reg, regularization_dict, displacement_dict, param_path)
```
"""
function make_regmap_matrix(centroid_dist_dict::Dict, roi_overlaps::Dict, q_dict::Dict, best_reg::Dict, regularization_dict::Dict, displacement_dict::Dict, 
        param::Dict, param_path::Dict; max_fixed_t::Int=0)
    return make_regmap_matrix(centroid_dist_dict, roi_overlaps, q_dict, best_reg, regularization_dict, displacement_dict, param_path,
        overlap_weight=param["overlap_weight"],
        centroid_weight=param["centroid_weight"],
        activity_diff_weight=param["activity_diff_weight"],
        q_weight=param["q_weight"],
        regularization_weight=param["regularization_weight"],
        displacement_weight=param["displacement_weight"],
        metric=param["quality_metric"],
        regularization_key=get(param, "regularization_key", "nonrigid_penalty"),
        self_weight=param["matrix_self_weight"],
        max_fixed_t=max_fixed_t,
        max_dist=param["max_centroid_dist"],
        min_weight=param["min_cluster_weight"]
    )
end

"""
    pairwise_dist(regmap_matrix; threshold::Real=1e-16, dtype::Type=Float64)

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
    r, c, v = findnz(d)
    for x=1:length(r)
        i = r[x]
        j = c[x]
        if i > j
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
    delete_smeared_neurons(roi, threshold)

Deletes neurons that might come from huge deformations in registration from an `roi` (neurons that have more pixels than `threshold`).
"""
function delete_smeared_neurons(roi, threshold)
    roi_return = deepcopy(roi)
    list_roi_id = sort(unique(roi))
    list_roi_id = list_roi_id[2:end] # remove roi == 0
    
    for roi_id = list_roi_id
        if sum(roi_return .== roi_id) > threshold
            roi_return[roi_return .== roi_id] .= 0
        end
    end
        
    roi_return
end

"""
    update_label_map(label_map, matches)

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
    invert_label_map(label_map)

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
    find_neurons(regmap_matrix, label_map; overlap_threshold::Real=0.005, height_threshold::Real=-0.01, dtype::Type=Float64, pair_match::Bool=false)

Groups ROIs (Regions of Interest) into neurons based on a matrix of pairwise overlaps and a given label map.

# Arguments:
- `regmap_matrix`: Matrix representing the pairwise overlaps between the ROIs.
- `label_map`: A dictionary mapping original ROIs to new ROI labels for each time point.
- `overlap_threshold::Real`: A threshold for the fraction of overlapping ROIs from the same time point. Clusters with an overlap exceeding this threshold will not be merged. Default is 0.005.
- `height_threshold::Real`: The maximum distance or overlap between two ROIs that can be merged. Default is -0.01.
- `dtype::Type`: The desired data type for internal computations. Default is Float64.
- `pair_match::Bool`: A flag indicating whether to merge clusters with a maximum size of 2. Default is false.

# Returns:
- `new_label_map`: A dictionary mapping original ROIs to neuron labels for each time point.
- `new_inv_map`: A dictionary mapping neuron labels back to the original ROIs for each time point.

# Note:
This function uses hierarchical clustering with constraints to group ROIs into neurons. The provided label map is crucial for this process as it dictates the initial mapping of ROIs.
"""
function find_neurons(regmap_matrix, label_map; overlap_threshold::Real=0.05, height_threshold::Real=-0.0003, dtype::Type=Float64, pair_match::Bool=false)
    inv_map = invert_label_map(label_map)
    dist = -regmap_matrix
    clusters = hclust_minimum_threshold_sparse(dist, inv_map, overlap_threshold, height_threshold, pair_match=pair_match)
    
    n = length(keys(inv_map))
    c_to_roi = Dict()
    n_to_c = [find(clusters, i) for i in 1:n]
    
    new_label_map = update_label_map(label_map, n_to_c)
    new_inv_map = invert_label_map(new_label_map)
    return (new_label_map, new_inv_map)
end

"""
    find_neurons(regmap_matrix, label_map::Dict, param::Dict)

Groups ROIs into neurons based on a matrix of pairwise overlaps and a given label map.

# Arguments:
- `regmap_matrix`: Matrix of pairwise overlaps between the ROIs.
- `label_map::Dict`: Dictionary of dictionaries mapping original ROIs to new ROI labels, for each time point.
- `param::Dict`: Dictionary containing `cluster_overlap_thresh` and `cluster_height_thresh` parameter settings to use for clustering.

# Returns:
- `new_label_map`: Dictionary of dictionaries mapping original ROIs to neuron labels, for each time point.
- `inv_map`: Dictionary of dictionaries mapping time points to original ROIs, for each neuron label.
"""
function find_neurons(regmap_matrix, label_map::Dict, param::Dict)
    return find_neurons(regmap_matrix, label_map,
        overlap_threshold=param["cluster_overlap_thresh"],
        height_threshold=param["cluster_height_thresh"])
end

"""
    match_neurons_across_datasets(label_map_1::Dict, label_map_2::Dict, inv_map_reg::Dict, max_fixed_t::Int)

Matches neurons across multiple datasets.

# Arguments
- `label_map_1::Dict`: Label map of ROI/time points to neuron ID for dataset 1.
- `label_map_2::Dict`: Label map of ROI/time points to neuron ID for dataset 2.
- `inv_map-reg::Dict`: Map of neuron ID to ROI/time points for the registration between datasets. Time points in dataset 2 are shifted by `max_fixed_t`.
- `max_fixed_t::Int`: Maximum time point in the first dataset.
"""
function match_neurons_across_datasets(label_map_1::Dict, label_map_2::Dict, inv_map_reg::Dict, max_fixed_t::Int)
    matches_12 = Dict()
    matches_21 = Dict()
    @showprogress for neuron in keys(inv_map_reg)
        if length(keys(inv_map_reg[neuron])) < 2
            continue
        end
        # we have a match between a neuron in each frame
        neuron1 = []
        neuron2 = []
        for t in keys(inv_map_reg[neuron])
            if t <= max_fixed_t
                if inv_map_reg[neuron][t][1] in keys(label_map_1[t])
                    append!(neuron1, label_map_1[t][inv_map_reg[neuron][t][1]])
                end
            elseif inv_map_reg[neuron][t][1] in keys(label_map_2[t - max_fixed_t])
                append!(neuron2, label_map_2[t - max_fixed_t][inv_map_reg[neuron][t][1]])
            end
        end
        if length(neuron1) == 0 || length(neuron2) == 0
            continue
        end
        for n1 in neuron1, n2 in neuron2
            if !(n1 in keys(matches_12))
                matches_12[n1] = Dict()
            end
            if !(n2 in keys(matches_12[n1]))
                matches_12[n1][n2] = 0
            end
            matches_12[n1][n2] += 1
            if !(n2 in keys(matches_21))
                matches_21[n2] = Dict()
            end
            if !(n1 in keys(matches_21[n2]))
                matches_21[n2][n1] = 0
            end
            matches_21[n2][n1] += 1
        end
    end
    return (matches_12, matches_21)
end

"""
    register_immobilized_rois(regmap_matrix, label_map_regmap, inv_map_regmap, label_map_freelymoving, valid_rois_freelymoving, param, reg_timept)

Registers immobilized ROIs to the freely moving dataset.

# Arguments:
- `regmap_matrix`: Matrix of distances between the ROIs
- `label_map_regmap::Dict`: Dictionary of dictionaries mapping original ROIs to new ROI labels for the freely-moving-to-immobilized registration.
- `inv_map_regmap::Dict`: Dictionary of dictionaries mapping new ROI labels to original ROIs, for the freely-moving-to-immobilized registration.
- `label_map_freelymoving::Dict`: Dictionary of dictionaries mapping original ROIs to new ROI labels in the freely-moving dataset (one of the outputs of `find_neurons`).
- `valid_rois_freelymoving`: List of which ROIs correspond to neurons in the freely-moving dataset.
- `param::Dict`: Dictionary containing `max_t` parameter setting to use for registration.
- `reg_timept`: Time point of the immobilized dataset to register.

# Returns:
- `roi_matches`: Dictionary of dictionaries mapping immobilized ROIs to matched freely moving ROIs.
- `inv_matches`: Dictionary of dictionaries mapping matched freely moving ROIs to immobilized ROIs.
- `roi_matches_best`: Dictionary of dictionaries mapping immobilized ROIs to matched freely moving ROIs, with only the best match for each immobilized ROI.
- `roi_match_confidence`: Dictionary of dictionaries mapping immobilized ROIs to the confidence of the match.
"""
function register_immobilized_rois(regmap_matrix, label_map_regmap, inv_map_regmap, label_map_freelymoving,
        valid_rois_freelymoving, param, reg_timept)

    s = size(regmap_matrix)
    roi_matches = Dict()
    inv_matches = Dict()
    t_imm = param["max_t"] + reg_timept

    for roi_imm in keys(label_map_regmap[t_imm])
        roi_matches[roi_imm] = Dict()
        roi_imm_label = label_map_regmap[t_imm][roi_imm]
        for roi_moving_label = 1:s[2]
            if (regmap_matrix[roi_imm_label, roi_moving_label] <= 0) || roi_moving_label == roi_imm_label
                continue
            end
            dict_moving = inv_map_regmap[roi_moving_label]
            @assert(length(keys(dict_moving)) == 1, "Inverse regmap matrix map must be 1:1 in time.")
            t_moving = collect(keys(dict_moving))[1]
            @assert(length(collect(values(dict_moving))[1]) == 1, "Inverse regmap matrix must be 1:1 in ROIs.")
            roi_moving = collect(values(dict_moving))[1][1]
            
            if !(t_moving in keys(label_map_freelymoving)) ||
                    !(roi_moving in keys(label_map_freelymoving[t_moving]))
                continue
            end
            
            neuron_label_moving = label_map_freelymoving[t_moving][roi_moving]
            neuron_moving = findall(n->n==neuron_label_moving, valid_rois_freelymoving)

            # the ROI we registered to was not succesfully mapped in freely-moving dataset
            if length(neuron_moving) == 0
                continue
            end
            
            @assert(length(neuron_moving)==1, "Number of neurons per label must be 1.")
            neuron_moving = neuron_moving[1]

            if neuron_moving in keys(roi_matches[roi_imm])
                roi_matches[roi_imm][neuron_moving] += regmap_matrix[roi_imm_label, roi_moving_label]
            else
                roi_matches[roi_imm][neuron_moving] = regmap_matrix[roi_imm_label, roi_moving_label]
            end

            if !(neuron_moving in keys(inv_matches))
                inv_matches[neuron_moving] = Dict()
            end
            if roi_imm in keys(inv_matches[neuron_moving])
                inv_matches[neuron_moving][roi_imm] += regmap_matrix[roi_imm_label, roi_moving_label]
            else
                inv_matches[neuron_moving][roi_imm] = regmap_matrix[roi_imm_label, roi_moving_label]
            end
        end
    end

    roi_match_best = zeros(Int, maximum(keys(roi_matches)))
    roi_match_confidence = zeros(maximum(keys(roi_matches)))
    for roi in keys(roi_matches)
        if length(keys(roi_matches[roi])) == 0
            continue
        end
        s1 = sum(values(roi_matches[roi]))
        for neuron in keys(roi_matches[roi])
            s2 = sum(values(inv_matches[neuron])) 
            confidence1 = 2 * roi_matches[roi][neuron] - s1
            confidence2 = 2 * inv_matches[neuron][roi] - s2
            confidence = min(confidence1, confidence2)
            if confidence > 0
                @assert(roi_match_best[roi] == 0, "Cannot have multiple confident matches")
                roi_match_best[roi] = Int(neuron)
                roi_match_confidence[roi] = confidence
            end
        end
    end
    roi_matches, inv_matches, roi_match_best, roi_match_confidence
end
