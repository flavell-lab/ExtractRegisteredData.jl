"""
Runs transformix to map centroids from the fixed frame to the moving frame.
Note that this is the opposite of `run_transformix_roi`, because of the way `transformix` is implemented.

# Arguments

`path::String`: Path to directory containing elastix registration
`output::String`: Output file directory
`centroids`: Centroids to transform
`parameters::String`: Transform parameter file
`transformix_dir::String`: Path to `transformix` executable
"""
function run_transformix_centroids(path, output, centroids, parameters, transformix_dir)
    curr_path = pwd()
    cd(path)
    cmd = Cmd([transformix_dir, "-out $output", "-def $centroids", "-tp $parameters"])
    result = run(cmd |> "transformix.log")
    cd(curr_path)
    return result
end

"""
Runs transformix to map an image from the moving frame to the fixed frame. 
Returns the resulting transformed image.
Note that this is the opposite of `run_transformix_centroids`, because of the way `transformix` is implemented.

`path::String`: Path to directory containing elastix registration
`output::String`: Output file directory
`input::String`: Input file path
`parameters::String`: Transform parameter file
`transformix_dir::String`: Path to `transformix` executable
"""
function run_transformix_img(path::String, output::String, input::String, parameters::String, transformix_dir::String)
    create_dir(output)
    cmd = Cmd(Cmd([transformix_dir, "-out", output, "-in", input, "-tp", parameters]), dir=path)
    result = read(cmd)
    return (read_img(MHD(joinpath(output, "result.mhd"))), result)
end


"""
Runs transformix to map an ROI image from the moving frame to the fixed frame. 
Modifies various transform parameters to force nearest-neighbors interpolation.
Returns the resulting transformed image.
Note that this is the opposite of `run_transformix_centroids`, because of the way `transformix` is implemented.

`path::String`: Path to directory containing elastix registration. Other paths should be absolute; they are NOT relative to this.
`input::String`: Input file path
`output::String`: Output file directory path
`parameters::String`: Path to transform parameter file
`parameters_roi::String`: Filename to save modified parameters.
`transformix_dir::String`: Path to `transformix` executable
"""
function run_transformix_roi(path::String, input::String, output::String, parameters::String, parameters_roi::String, transformix_dir::String)
    create_dir(output)
    modify_parameter_file(parameters, parameters_roi, Dict("FinalBSplineInterpolationOrder" => 0, "DefaultPixelvalue" => 0))
    cmd = Cmd(Cmd([transformix_dir, "-out", output, "-in", input, "-tp", parameters_roi]), dir=path)
    result = read(cmd)
    return (read_img(MHD(joinpath(output, "result.mhd"))), result)
end


"""
Runs transformix on each registration problem in a graph. Chains registrations together,
such that the root node is mapped to each other frame, where the data could then be extracted.

# Arguments

- `rootpath::String`: Working directory; all other directories are relative to this
- `root_node_roi::String`: Path to root node ROI file (relative to `rootpath`)
- `out_dir::String`: Output directory. Result of root node mapped to frame i will be stored in `root_path/out_dir/i/result.mhd`
- `graph::SimpleWeightedDiGraph`: Graph of registration problems (this is the subgraph of problems that were done, not the full difficulty graph)
- `root_node::Integer`: Root node.
- `best_reg::Dict`: Dictionary that maps each registration problem (i,j) to the best registration for that problem.
    Eg if (i,j) maps to (x,y), then `rootpath/regdir/itoj/TransformParameters.x.Ry.txt` will be used.
- `transformix_dir::String`: ABSOLUTE path to `transformix` executable.

# Optional keyword arguments

- `regdir::String`: Directory to registrations. Default Registered.
"""
function run_transformix_roi_graph(rootpath::String, root_node_roi::String, out_dir::String,
    graph::SimpleWeightedDiGraph, root_node::Integer, best_reg::Dict, transformix_dir::String; regdir::String="Registered")
    # run through graph in order
    eds = []
    curr_vertices = Queue{Integer}()
    enqueue!(curr_vertices, root_node)
    while !isempty(curr_vertices)
        vertex = dequeue!(curr_vertices)
        for edge in edges(graph)
            if vertex == src(edge)
                push!(eds, (vertex, dst(edge)))
                if !(dst(edge) in curr_vertices)
                    enqueue!(curr_vertices, dst(edge))
                end
            end
        end
    end

    n = length(eds)
    @showprogress for i = 1:n
        e = eds[i]
        create_dir(joinpath(rootpath, out_dir, string(e[1])))
        if e[1] == root_node
            substitutions = Dict()
            substitutions["ElementDataFile"] = "result.raw"
            modify_mhd(joinpath(rootpath, root_node_roi), joinpath(rootpath, out_dir, string(e[1]), "result.mhd"), substitutions)
            cp(joinpath(rootpath, root_node_roi), joinpath(rootpath, out_dir, string(e[1]), "result.raw"), force=true)
        end
        input = joinpath(rootpath, out_dir, string(e[1]), "result.mhd")
        run_transformix_roi(joinpath(rootpath, regdir, "$(e[1])to$(e[2])"), input,
                joinpath(rootpath, out_dir, string(e[2])),
                joinpath(rootpath, regdir, "$(e[1])to$(e[2])", "TransformParameters.$(best_reg[e][1]).R$(best_reg[e][2]).txt"),
                joinpath(rootpath, regdir, "$(e[1])to$(e[2])", "TransformParameters.$(best_reg[e][1]).R$(best_reg[e][2])_roi.txt"),
                transformix_dir)
    end
end
