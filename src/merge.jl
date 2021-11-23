function merge_confocal_data!(combined_data_dict::Dict, data_dict::Dict, data_dict_2::Dict, dataset::String; manual_remap::Bool=false)
    data_dict["roi_match_$dataset"] = zeros(Int32, length(data_dict["valid_rois"]))
    for i in 1:length(data_dict["valid_rois"])
        if isnothing(data_dict["valid_rois"][i])
            continue
        end
        for k in keys(data_dict["matches_to_$dataset"][data_dict["valid_rois"][i]])
            if k in data_dict_2["valid_rois"] && data_dict["matches_to_$dataset"][data_dict["valid_rois"][i]][k] > 400
                data_dict["roi_match_$dataset"][i] = findall(x->x==k, data_dict_2["valid_rois"])[1]
            end
        end
    end

    data_dict["successful_idx_$dataset"] = [i for i in 1:length(data_dict["valid_rois"]) if (data_dict["roi_match_$dataset"][i] != 0)]
    max_t_all = size(data_dict["zscored_traces_array"],2) + size(data_dict_2["deconvolved_traces_array"],2)
    combined_data_dict["deconvolved_traces_array"] = zeros(length(data_dict["successful_idx_$dataset"]), max_t_all)
    combined_data_dict["deconvolved_traces_array"][:,1:data_dict["max_t"]] .= data_dict["deconvolved_traces_array"][data_dict["successful_idx_$dataset"],:]
    mean_val_1 = mean(data_dict["deconvolved_traces_array"][data_dict["successful_idx_$dataset"],:])
    mean_val_2 = mean(data_dict_2["deconvolved_traces_array"][data_dict["roi_match_$dataset"][data_dict["successful_idx_$dataset"]],:])
    combined_data_dict["deconvolved_traces_array"][:,data_dict["max_t"]+1:end] .= mean_val_1 / mean_val_2 .* data_dict_2["deconvolved_traces_array"][data_dict["roi_match_$dataset"][data_dict["successful_idx_$dataset"]],:];

    combined_data_dict["zscored_traces_array"] = zeros(size(combined_data_dict["deconvolved_traces_array"]))
    for n=1:size(combined_data_dict["deconvolved_traces_array"],1)
        combined_data_dict["zscored_traces_array"][n,:] .= zscore(combined_data_dict["deconvolved_traces_array"][n,:])
    end

    # these datasets had manual remapping
    inverse_map = []
    for i=1:maximum(data_dict["successful_idx_$dataset"])
        if i in data_dict["successful_idx_$dataset"]
            push!(inverse_map, findall(x->x==i, data_dict["successful_idx_$dataset"])[1])
        else
            push!(inverse_map, NaN)
        end
    end

    combined_data_dict["timestamps"] = deepcopy(data_dict["timestamps"])
    append!(combined_data_dict["timestamps"], data_dict_2["timestamps"])
    combined_data_dict["timestamps"] = combined_data_dict["timestamps"] .- combined_data_dict["timestamps"][1] 
    
    combined_data_dict["nir_timestamps"] = deepcopy(data_dict["nir_timestamps"])
    append!(combined_data_dict["nir_timestamps"], data_dict_2["nir_timestamps"])
    combined_data_dict["nir_timestamps"] = combined_data_dict["nir_timestamps"] .- combined_data_dict["nir_timestamps"][1];

    combined_data_dict["stim_begin_nir"] = deepcopy(data_dict["stim_begin_nir"])
    append!(combined_data_dict["stim_begin_nir"], data_dict_2["stim_begin_nir"])
    combined_data_dict["stim_begin_confocal"] = deepcopy(data_dict["stim_begin_confocal"])
    append!(combined_data_dict["stim_begin_confocal"], data_dict_2["stim_begin_confocal"])

        
    valid_rois_remapped = [isnan(x) ? nothing : data_dict["valid_rois"][x] for x in inverse_map]
    combined_data_dict["valid_rois"] = valid_rois_remapped
    combined_data_dict["inv_map"] = data_dict["inv_map"]

    combined_data_dict["num_neurons"] = size(combined_data_dict["zscored_traces_array"], 1)


    # there should not be any bad neurons unless neurons were manually remapped
    zscored_traces_nobadneurons = zeros(size(combined_data_dict["zscored_traces_array"]))
    count = 1
    for n in 1:combined_data_dict["num_neurons"]
        if isnan(combined_data_dict["zscored_traces_array"][n,1])
            continue
        end
        zscored_traces_nobadneurons[count,:] .= combined_data_dict["zscored_traces_array"][n,:]
        count += 1
    end
    combined_data_dict["M_z"], combined_data_dict["X_z"], combined_data_dict["Yt_z"] = multivar_fit(zscored_traces_nobadneurons[1:count-1,:], PCA, maxoutdim=20);

    max_t_all = size(data_dict["zscored_traces_array"],2) + size(data_dict_2["zscored_traces_array"],2)
    combined_data_dict["max_t"] = max_t_all
    combined_data_dict["max_t_1"] = data_dict["max_t"];
end
