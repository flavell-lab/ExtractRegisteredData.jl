# ExtractRegisteredData.jl

## Prerequisites

This package requires you to have previously installed the `FlavellBase.jl`, `ImageDataIO.jl`, `MHDIO.jl`, `CaSegmentation.jl`, and `SegmentationTools.jl` packages.
It also requires you to have [installed `transformix`](https://simpleelastix.readthedocs.io/GettingStarted.html#manually-building-on-linux).

## Running `transformix` on ROIs

In order to extract traces from data registered with `elastix`, you need to run `transformix` on the ROIs of interest. This is important both for aligning the green and red channels (and thereby ensuring traces are extracted from the correct region in the green channel), and for aligning the red channels in different frames through registration.

This is all accomplished with the `run_transformix_roi` function:

```julia
# img will now contain the ROI image mapped to the new frame via registration
# result will contain the output from `transformix` for debugging purposes
img, result = run_transformix_roi("/path/to/data", "root_node_roi.mhd", "transformed_roi", "/path/to/transform/file", "/path/to/save/modified/transform/file", "/path/to/transformix")
```

## Neuron matching

To match ROIs between frames - and thereby identify neurons - the `register_neurons_overlap` function finds overlaps between ROIs across two frames. Assuming you've previously run `transformix` on the moving ROI image to get the transformed image `img` (with the fixed ROI image `roi_fixed`), and obtained red channel activity for both images:

```julia
# delete neurons that were overly smeared by registration
roi_moving = delete_smeared_neurons(img)
roi_overlap, roi_activity = register_neurons_overlap(roi_moving, roi_fixed, moving_activity, fixed_activity)
```

By doing this for every pair of registered frames, we can obtain dictionaries `roi_overlaps` and `roi_activity_diff`. Once we have these dictionaries, together with the outputs `q_dict` and `best_reg` of `make_quality_dict` (from the `RegistrationGraph.jl` package), and a dictionary `watershed_errors` of lists of ROIs with segementation errors for every frame, we can make a matrix of all ROIs across all frames, whose `(i,j)`th entry is the quality of the match between ROIs `i` and `j`. There is also `label_map` which is a dictionary mapping frames and original ROIs to these new matrix-index ROIs.

From here, we can cluster the matrix to get a map of ROIs to neurons, and vice versa (**WARNING: the clustering method currently uses an extremely large amount of memory. It is recommended to set the data type to Float16 until this is fixed.**)

```julia
regmap_matrix, label_map = make_regmap_matrix(roi_overlaps, roi_activity_diff, q_dict, best_reg, watershed_errors)

new_label_map, inv_map = find_neurons(regmap_matrix, label_map; dtype=Float16)
```

## Extracting traces

Once we have the neuron identity maps, we can extract traces, which will be stored as a dictionary mapping neurons to dictionaries mapping time points to the neural activity. Note that many of the traces will be for neurons that were only detected in a very small number of frames - this usually happens if a neuron is difficult to register, or is inconsistently detected by the UNet. It is recommended to threshold to remove neurons that are not present in many frames.

```julia
traces = extract_traces(inv_map, "/path/to/gcamp/activities")
```
