function merge_confocal_data!(combined_data_dict::Dict, data_dict::Dict, data_dict_2::Dict, dataset::String; manual_remap::Bool=false)
    # these datasets had manual remapping
    inverse_map = []
    if manual_remap
        for i=1:maximum(data_dict["roi_self_match_arr"])
            if i in data_dict["roi_self_match_arr"]
                push!(inverse_map, findall(x->x==i, data_dict["roi_self_match_arr"])[1])
            else
                push!(inverse_map, NaN)
            end
        end
    else
        for i=1:maximum(data_dict["successful_idx_$dataset"])
            if i in data_dict["successful_idx_$dataset"]
                push!(inverse_map, findall(x->x==i, data_dict["successful_idx_$dataset"])[1])
            else
                push!(inverse_map, NaN)
            end
        end
    end

    combined_data_dict["timestamps"] = deepcopy(data_dict["timestamps"])
    append!(combined_data_dict["timestamps"], other_dataset_data_dict[dataset]["timestamps"])
    combined_data_dict["timestamps"] = combined_data_dict["timestamps"] .- combined_data_dict["timestamps"][1] 
    
    combined_data_dict["nir_timestamps"] = deepcopy(data_dict["nir_timestamps"])
    append!(combined_data_dict["nir_timestamps"], other_dataset_data_dict[dataset]["nir_timestamps"])
    combined_data_dict["nir_timestamps"] = combined_data_dict["nir_timestamps"] .- combined_data_dict["nir_timestamps"][1];

    combined_data_dict["stim_begin_nir"] = deepcopy(data_dict["stim_begin_nir"])
    append!(combined_data_dict["stim_begin_nir"], other_dataset_data_dict[dataset]["stim_begin_nir"])
    combined_data_dict["stim_begin_confocal"] = deepcopy(data_dict["stim_begin_confocal"])
    append!(combined_data_dict["stim_begin_confocal"], other_dataset_data_dict[dataset]["stim_begin_confocal"])

        
    valid_rois_remapped = [isnan(x) ? nothing : data_dict["valid_rois"][x] for x in inverse_map]
    combined_data_dict["valid_rois"] = valid_rois_remapped
    combined_data_dict["inv_map"] = data_dict["inv_map"]


    # there should not be any bad neurons unless neurons were manually remapped
    zscored_traces_nobadneurons = zeros(size(combined_data_dict["zscored_traces_array"]))
    count = 1
    for n in 1:data_dict["num_neurons"]
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
