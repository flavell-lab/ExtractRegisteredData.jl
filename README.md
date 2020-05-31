# ExtractRegisteredData.jl

## Prerequisites

This package requires you to have previously installed the `FlavellBase.jl`, `ImageDataIO.jl`, `MHDIO.jl`, `CaSegmentation.jl`, and `SegmentationTools.jl` packages.
It also requires you to have [installed `transformix`](https://simpleelastix.readthedocs.io/GettingStarted.html#manually-building-on-linux).

## Running `transformix` on ROIs

In order to extract traces from data registered with `elastix`, you need to run `transformix` on the ROIs of interest.
This is accomplished with the `run_transformix_roi_graph` function; it assumes you've previously generated a registration graph `subgraph` 
and a dictionary of best registrations `best_reg`:

```julia
# first map the central frame's ROIs to each other frame
run_transformix_roi_graph("/path/to/data", "root_node_roi.mhd", "transformed_roi", subgraph, best_reg, "/path/to/transformix")

# then extract traces from all frames
data = extract_traces_roi(1:100, "/path/to/data", "transformed_roi", "MHD", "img_prefix", 2)
```

`data` will now contain a dictionary that maps each frame to an array containing intensity values for each ROI.
An activity value of 0 means the ROI moved out of the field of view.
