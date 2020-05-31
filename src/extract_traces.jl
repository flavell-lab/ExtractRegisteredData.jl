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
