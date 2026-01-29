module LLMBenchSimple

export @bench, setup_problem, grade, list_problems, split_problem_id,
       llmbench_workdir, mktempdir_bench, get_commit_patch, compare_commit_with_file_patch, get_commit_diff, compare_diffs,
       get_bash_uid, unprivileged, chown_unprivileged

using Test
using Base.ScopedValues: ScopedValue, @with
using LibGit2
using Base.Meta

# Include the improved testset handling
include("testset_handling.jl")

# Include patch utilities
include("patch_utils.jl")

# Compatibility layer for Julia 1.11 vs 1.13+
# LazyScopedValue and OncePerProcess were added in Julia 1.13 (PR #59372)
if isdefined(Base.ScopedValues, :LazyScopedValue) && isdefined(Base, :OncePerProcess)
    # Julia 1.13+: use the new LazyScopedValue API
    using Base.ScopedValues: LazyScopedValue
    using Base: OncePerProcess

    const WORKDIR_CONTEXT = LazyScopedValue{Union{String,Nothing}}(
        OncePerProcess{Union{String,Nothing}}() do
            get(ENV, "LLMBENCH_WORKSPACE", nothing)
        end
    )
else
    # Julia 1.11-1.12: use regular ScopedValue with default value
    # Initialization will happen on first access via llmbench_workdir()
    const WORKDIR_CONTEXT = ScopedValue{Union{String,Nothing}}(nothing)
end

# Debug scoped value - when true, temp directories are persisted and logged
const DEBUG_BENCH = ScopedValue{Bool}(false)

# Strip module name prefix if present (e.g., "module_name-problem_id" -> "problem_id")
function split_problem_id(mod::Module, problem_id::String)
    if !isempty(problem_id)
        module_name = string(nameof(mod))
        prefix = module_name * "-"
        if startswith(problem_id, prefix)
            problem_id = problem_id[length(prefix)+1:end]
            @debug "Stripped prefix" original_problem_id problem_id prefix
        end
    end
    return problem_id
end

# Transform grading result into standardized score dictionary
function result_to_score_dict(result, problem_id::String)
    # Handle different result types
    if result isa Dict
        # Already in dict format, ensure it has the required structure
        if haskey(result, "score")
            return result
        else
            # Create standard format
            score = get(result, "success", false) ? 1.0 : 0.0
            return Dict(
                "subscores" => Dict(problem_id => score),
                "weights" => Dict(problem_id => 1.0),
                "score" => score
            )
        end
    elseif result isa Bool
        # Boolean result - true = success, false = failure
        score = result ? 1.0 : 0.0
        return Dict(
            "subscores" => Dict(problem_id => score),
            "weights" => Dict(problem_id => 1.0),
            "score" => score
        )
    elseif isdefined(Test, :Pass) && result isa Test.Pass
        # Test.Pass result - success
        return Dict(
            "subscores" => Dict(problem_id => 1.0),
            "weights" => Dict(problem_id => 1.0),
            "score" => 1.0
        )
    elseif isdefined(Test, :Fail) && result isa Test.Fail
        # Test.Fail result - failure
        return Dict(
            "subscores" => Dict(problem_id => 0.0),
            "weights" => Dict(problem_id => 1.0),
            "score" => 0.0,
            "metadata" => Dict("error" => string(result))
        )
    elseif isdefined(Test, :Error) && result isa Test.Error
        # Test.Error result - error occurred
        return Dict(
            "subscores" => Dict(problem_id => 0.0),
            "weights" => Dict(problem_id => 1.0),
            "score" => 0.0,
            "metadata" => Dict("error" => sprint(showerror, result.value, result.backtrace))
        )
    elseif result isa Float64 || result isa Float32 || result isa Float16
        # Floating point score directly
        score = Float64(result)
        # Clamp to [0, 1] range
        score = clamp(score, 0.0, 1.0)
        return Dict(
            "subscores" => Dict(problem_id => score),
            "weights" => Dict(problem_id => 1.0),
            "score" => score
        )
    else
        # Unknown result type - throw error instead of returning 0
        error("Unknown result type in grader: $(typeof(result)). Expected Dict, Bool, Float64, Test.Pass, Test.Fail, or Test.Error")
    end
end

# Function to get the current workdir from context
function llmbench_workdir()
    workdir = WORKDIR_CONTEXT[]

    # For Julia 1.11-1.12 compatibility: check environment if workdir is not set
    if workdir === nothing && !isdefined(Base.ScopedValues, :LazyScopedValue)
        # Fallback: try to get from environment variable
        env_workdir = get(ENV, "LLMBENCH_WORKSPACE", nothing)
        if env_workdir !== nothing
            return env_workdir
        end
    end

    if workdir === nothing
        error("llmbench_workdir() can only be called within a benchmark setup or grading context, or with LLMBENCH_WORKSPACE environment variable set")
    end

    return workdir
end

# Helper function to run code with workdir context
function with_workdir(f, workdir::String)
    @Base.ScopedValues.with WORKDIR_CONTEXT => abspath(workdir) begin
        f()
    end
end

# Benchmark-aware temporary directory creation
# If DEBUG_BENCH is enabled, the temp directory is persisted and logged
# Otherwise, behaves like standard mktempdir with automatic cleanup
function mktempdir_bench(f)
    if DEBUG_BENCH[]
        # Create temp dir without automatic cleanup
        tmpdir = mktempdir()
        @info "Debug: Temporary directory created and persisted" tmpdir
        try
            return f(tmpdir)
        catch e
            # Don't cleanup even on error when debugging
            rethrow(e)
        end
    else
        # Standard mktempdir with automatic cleanup
        mktempdir(f)
    end
end

const all_macros = (:promptval, :promptdir, :promptcode, :promptcommit, :promptmd)
for mname in all_macros
    @eval begin
        macro $(Symbol(string(mname, "_str")))(s)
            error(string($(string(mname)), "\"...\" can only be used inside @bench"))
        end
        macro $(Symbol(string("raw", mname, "_str")))(s)
            error(string("raw", $(string(mname)), "\"...\" can only be used inside @bench"))
        end
        export $(Symbol(string("@", mname, "_str"))), $(Symbol(string("@raw", mname, "_str")))
    end
end

# Helper functions for the new @bench macro
function make_prompt_expr(prompt_data)
    expr = copy(prompt_data)
    @assert isexpr(expr, :macrocall)
    name = expr.args[1]
    prompt_str = expr.args[3]
    
    # Check if it's a raw prompt (no interpolation)
    is_raw = occursin("raw", string(name))
    
    if is_raw
        # For raw prompts, just pass the string directly
        :(LLMBenchSimple.make_prompt($(quot(name)), $prompt_str))
    else
        # For regular prompts with interpolation, parse at runtime
        # The prompt_str might contain interpolations like $x
        :(LLMBenchSimple.make_prompt($(quot(name)), $prompt_str))
    end
end

function make_answer_extract(prompt_data)
    expr = copy(prompt_data)
    @assert isexpr(expr, :macrocall)
    name = expr.args[1]
    :(LLMBenchSimple.extract_answer($(quot(name)), workdir, transcript))
end

macro_variants(name) = (Symbol(string("@", name)), Symbol(string("@raw", name)),
                        Symbol(string("@", name, "_str")), Symbol(string("@raw", name, "_str")))

# Generate appropriate prompt text based on prompt type
function make_prompt(prompt_type::Symbol, md_content)
    # Convert the markdown content to string
    prompt_text = string(md_content)
    
    if prompt_type in macro_variants(:promptval)
        # For promptval, provide instructions about answer tags
        return """
        Problem: $prompt_text
        
        Please provide your answer wrapped in <answer> tags.
        For example: <answer>your_answer_here</answer>
        """
    elseif prompt_type in macro_variants(:promptdir)
        # For promptdir, return instructions about workspace directory
        workdir = llmbench_workdir()
        return """
        Task: $prompt_text
        
        Please complete this task in the workspace directory: $workdir
        Create any necessary files and directories in this location.
        """
    elseif prompt_type in macro_variants(:promptcode)
        # For promptcode, provide instructions about answer.jl file
        workdir = llmbench_workdir()
        return """
        Task: $prompt_text
        
        Please write your Julia code solution and save it to:
        $(joinpath(workdir, "answer.jl"))
        
        The code will be parsed as Julia code and evaluated.
        """
    elseif prompt_type in macro_variants(:promptcommit)
        # For promptcommit, provide instructions about git commit
        workdir = llmbench_workdir()
        return """
        Task: $prompt_text
        
        Please complete this task in the git repository at: $workdir
        Make changes and create a *single* git commit as specified.
        """
    elseif prompt_type in macro_variants(:promptmd)
        # For promptmd, provide instructions about answer.md file
        workdir = llmbench_workdir()
        return """
        Task: $prompt_text
        
        Please write your markdown solution and save it to:
        $(joinpath(workdir, "answer.md"))
        
        The markdown content will be read and returned as the answer.
        """
    else
        # Default fallback
        return prompt_text
    end
end

# Extract answer from transcript based on prompt type
function extract_answer(prompt_type::Symbol, workdir::String, transcript::String)
    if prompt_type in macro_variants(:promptval)
        # Extract answer from <answer> tags in transcript
        # Find all matches and use the last one (to avoid example tags)
        matches = collect(eachmatch(r"<answer>(.*?)</answer>"s, transcript))
        if !isempty(matches)
            answer_str = strip(matches[end].captures[1])
            # Try to parse the answer as Julia code
            try
                return Meta.parse(answer_str)
            catch
                # If parsing fails, return as string
                return answer_str
            end
        end
        return nothing
    elseif prompt_type in macro_variants(:promptdir)
        # For directory prompts, return the workspace directory path
        return workdir
    elseif prompt_type in macro_variants(:promptcode)
        # Read answer from answer.jl file
        answer_file = joinpath(workdir, "answer.jl")
        if isfile(answer_file)
            code_str = read(answer_file, String)
            # Parse all expressions in the code string
            return Meta.parseall(code_str; filename="answer.jl")
        else
            return nothing
        end
    elseif prompt_type in macro_variants(:promptcommit)
        # For commit prompts, get the last commit
        if isdir(joinpath(workdir, ".git"))
            try
                repo = LibGit2.GitRepo(workdir)
                head = LibGit2.head(repo)
                commit = LibGit2.peel(LibGit2.GitCommit, head)
                return commit
            catch
                return nothing
            end
        end
        return nothing
    elseif prompt_type in macro_variants(:promptmd)
        # Read answer from answer.md file
        answer_file = joinpath(workdir, "answer.md")
        if isfile(answer_file)
            return read(answer_file, String)
        else
            return nothing
        end
    else
        # Default: try to extract from <answer> tags
        m = match(r"<answer>(.*?)</answer>"s, transcript)
        return m !== nothing ? strip(m.captures[1]) : nothing
    end
end

macro bench(args...)
    # Parse arguments: can be @bench "name" expr or @bench "name" [metadata...] expr
    local name, expr, metadata_pairs

    if length(args) == 2
        # @bench "name" expr
        name = Symbol(args[1])
        expr = args[2]
        metadata_pairs = []
    elseif length(args) == 3
        # @bench "name" [metadata...] expr
        name = Symbol(args[1])
        metadata_expr = args[2]
        expr = args[3]

        # Parse metadata from vector expression
        if isa(metadata_expr, Expr) && metadata_expr.head == :vect
            metadata_pairs = metadata_expr.args
        else
            return :(error("@bench metadata must be a vector expression like [key=value, ...]"))
        end
    else
        return :(error("@bench requires 2 or 3 arguments: @bench \"name\" [metadata...] expr"))
    end

    # Split into setup, prompt, and grading parts
    setup_expr, prompt_data, grading_expr = split_at_first_prompt(expr)

    if prompt_data === nothing
        return :(error($("@bench \"$name\" does not contain any prompt macros (promptval\"\", promptdir\"\", promptcode\"\", promptcommit\"\", or promptmd\"\"). At least one prompt macro is required.")))
    end

    prompt = make_prompt_expr(prompt_data)
    answer_extract = make_answer_extract(prompt_data)

    # Process metadata into a Dict expression
    metadata_dict_expr = :(Dict{String,Any}())
    for pair in metadata_pairs
        if isa(pair, Expr) && pair.head == :(=)
            key = string(pair.args[1])
            value = pair.args[2]
            push!(metadata_dict_expr.args, :($(key) => $(value)))
        else
            return :(error("@bench metadata must be key=value pairs"))
        end
    end

    # Generate code that checks at runtime if BENCHMARKS exists
    return esc(quote
        # Check if BENCHMARKS exists at runtime
        if !isdefined(@__MODULE__, :BENCHMARKS)
            const BENCHMARKS = Symbol[]
            const BENCHMARK_METADATA = Dict{Symbol,Dict{String,Any}}()

            # Define setup_problem function
            function setup_problem(workdir::String, problem_id::String="")
                if isempty(problem_id)
                    return "Error: problem_id is required. Available problems: " * join(BENCHMARKS, ", ")
                end
                
                # Strip module prefix if present
                clean_id = LLMBenchSimple.split_problem_id(@__MODULE__, problem_id)
                
                # Check if the method exists
                if !hasmethod(setup_problem, Tuple{String, Val{Symbol(clean_id)}})
                    return "Error: Problem '$clean_id' not found. Available problems: " * join(BENCHMARKS, ", ")
                end
                
                setup_problem(workdir, Val{Symbol(clean_id)}())
            end
            export setup_problem

            # Define grade function
            function grade(workdir::String, transcript::String, problem_id::String="")
                if isempty(problem_id)
                    return Dict("score" => 0.0, "metadata" => Dict("error" => "Error: problem_id is required. Available problems: " * join(BENCHMARKS, ", ")))
                end

                # Strip module prefix if present
                clean_id = LLMBenchSimple.split_problem_id(@__MODULE__, problem_id)

                # Check if the method exists
                if !hasmethod(grade, Tuple{String, String, Val{Symbol(clean_id)}})
                    return Dict("score" => 0.0, "metadata" => Dict("error" => "Error: Problem '$clean_id' not found. Available problems: " * join(BENCHMARKS, ", ")))
                end

                # The Val-based grade function now returns a score dict directly
                return grade(workdir, transcript, Val{Symbol(clean_id)}())
            end
            export grade

            # Define list_problems function
            function list_problems()
                # Return list of problems with metadata
                return [
                    merge(
                        Dict{String,Any}("id" => string(problem)),
                        get(BENCHMARK_METADATA, problem, Dict{String,Any}())
                    )
                    for problem in BENCHMARKS
                ]
            end
            export list_problems
        end

        function setup_problem(workdir::String, problem_id::Val{$(quot(name))})
            @Base.ScopedValues.with $(WORKDIR_CONTEXT) => abspath(workdir) begin
                $setup_expr
                $prompt
            end
        end

        function grade(workdir::String, transcript::String, problem_id::Val{$(quot(name))})
            @Base.ScopedValues.with $(WORKDIR_CONTEXT) => abspath(workdir) begin
                var"##answer##" = $answer_extract
                # Evaluate grading expression with test capture
                result_tuple = LLMBenchSimple.evaluate_with_test_capture() do
                    $grading_expr
                end
                
                # Unpack the tuple (result, error_info)
                result, error_info = result_tuple
                
                # Convert to score dict and add error info if present
                score_dict = LLMBenchSimple.result_to_score_dict(result, $(string(name)))
                if error_info !== nothing
                    if !haskey(score_dict, "metadata")
                        score_dict["metadata"] = Dict{String,Any}()
                    end
                    score_dict["metadata"]["test_errors"] = error_info
                end
                
                return score_dict
            end
        end

        push!(BENCHMARKS, $(quot(name)))
        BENCHMARK_METADATA[$(quot(name))] = $metadata_dict_expr

        nothing
    end)
end

function is_prompt_macro(expr)
    if expr.head == :macrocall && length(expr.args) >= 2
        macro_name = expr.args[1]
        # Check for both regular and raw prompt macros
        # Macros can appear as :@promptval or Symbol("@promptval_str") depending on context
        if macro_name in [macro_variants(:promptval)...;
                          macro_variants(:promptdir)...;
                          macro_variants(:promptcode)...;
                          macro_variants(:promptcommit)...;
                          macro_variants(:promptmd)...]
            return true
        end
    end
    return false
end

function extract_and_replace_prompt(expr)
    isa(expr, Expr) || return nothing
    if is_prompt_macro(expr)
        return (expr, :var"##answer##")
    end
    args = nothing
    for i = 1:length(expr.args)
        extracted = extract_and_replace_prompt(expr.args[i])
        if extracted !== nothing
            (prompt_data, replacement) = extracted
            ret = copy(expr)
            expr.args[i] = replacement
            return (prompt_data, expr)
        end
    end
    return nothing
end

# Split expression into setup, prompt, and grading parts
function split_at_first_prompt(expr)
    # For simple non-block expressions
    if !isa(expr, Expr) 
        return (nothing, nothing, expr)
    end
   
    if isexpr(expr, :call)
        extracted = extract_and_replace_prompt(expr)
        extracted === nothing && return (expr, nothing, nothing)
        return (nothing, extracted...)
    else isexpr(expr, :block)
        # Look for prompt in the block
        setup_exprs = Any[]
        prompt_data = nothing
        grading_exprs = Any[]
        found_prompt = false
        
        for arg in expr.args
            if !found_prompt
                extracted = extract_and_replace_prompt(arg)
                if extracted !== nothing
                    (prompt_data, arg) = extracted
                    found_prompt = true
                end
            end
            if !found_prompt
                # Before prompt - goes to setup
                push!(setup_exprs, arg)
            else
                # After prompt - goes to grading
                push!(grading_exprs, arg)
            end
        end
        
        setup = Expr(:block, setup_exprs...)
        grading = Expr(:block, grading_exprs...)
        return (setup, prompt_data, grading)
    end
    
    # Default case - no prompt found
    return (expr, nothing, nothing)
end


# Get UID from environment (for unprivileged execution)
function get_bash_uid()
    uid_str = get(ENV, "LLMBENCH_BASH_UID", nothing)
    if uid_str === nothing
        return nothing
    end
    # Let parse throw an error if the value is invalid
    return parse(Int, uid_str)
end

"""
    unprivileged(cmd::Cmd; uid::Union{Int,Nothing}=get_bash_uid())

Return a modified `Cmd` that will run with the specified UID. Uses the UID from 
LLMBENCH_BASH_UID environment variable by default.

Uses `Base.setuid` when available (Julia 1.13+), otherwise creates a 
`sudo`-wrapped command.

Returns a `Cmd` object that can be run with `run()`.

# Examples
```julia
# Get command with UID from environment
cmd = unprivileged(`ls -la`)
run(cmd)

# Get command with specific UID
cmd = unprivileged(`whoami`, uid=1000)
run(cmd)

# No privilege modification (returns original cmd)
cmd = unprivileged(`ls`, uid=nothing)
run(cmd)
```
"""
function unprivileged(cmd::Cmd; uid::Union{Int,Nothing}=get_bash_uid())
    if uid === nothing
        # No UID specified, return original command
        return cmd
    end
    
    # Check if native setuid is available (Julia 1.13+)
    # Based on PR #59420: https://github.com/JuliaLang/julia/pull/59420
    # The PR adds setuid as a function that modifies a Cmd object
    if isdefined(Base, :setuid)
        # Use the native setuid function to create a modified Cmd
        return Base.setuid(cmd, uid)
    else
        # Create sudo-wrapped command for older Julia versions
        return create_sudo_cmd(cmd, uid)
    end
end

# Helper function to get user's home directory using getpwuid
function get_user_home(uid::Int)
    # Use Libc.getpwuid to get user info - it returns a Libc.Passwd object or nothing
    # Convert to UInt as getpwuid expects unsigned
    passwd_info = Libc.getpwuid(UInt(uid))
    
    if passwd_info === nothing
        # Failed to get user info, return nothing
        return nothing
    end
    
    # Libc.getpwuid returns a Libc.Passwd object with homedir field already as a string
    return passwd_info.homedir
end

# Helper function to create sudo-wrapped command
function create_sudo_cmd(cmd::Cmd, uid::Int)
    # Get the command as a string using Base's shell escaping
    # shell_escape each part and join them
    escaped_parts = [Base.shell_escape(part) for part in cmd.exec]
    cmd_str = join(escaped_parts, " ")
    
    # Get current PATH to restore it inside sudo
    current_path = get(ENV, "PATH", "")
    
    # Get target user's home directory
    user_home = get_user_home(uid)
    
    # Build the sudo command
    # -E: preserve environment variables (but PATH is still reset by sudo)
    # -u #uid: run as the specified UID (# prefix tells sudo it's a UID not username)
    # env PATH=...: explicitly restore PATH inside sudo (sudo has special handling for PATH)
    # sh -c: use shell to run the command
    sudo_cmd = `sudo -E -u "#$uid" env PATH=$current_path sh -c $cmd_str`

    # Preserve environment variables from the original command
    if cmd.env !== nothing
        # setenv replaces all env vars, so we use it to carry forward cmd's env
        sudo_cmd = setenv(sudo_cmd, cmd.env)
    end
    
    # Add HOME environment variable (on top of any existing env)
    if user_home !== nothing
        sudo_cmd = addenv(sudo_cmd, "HOME" => user_home)
    end
    
    # Preserve the working directory if specified
    if cmd.dir != ""
        sudo_cmd = Cmd(sudo_cmd; dir=cmd.dir)
    end
    
    # Preserve ignorestatus flag if set
    if cmd.ignorestatus
        sudo_cmd = Cmd(sudo_cmd; ignorestatus=true)
    end
    
    return sudo_cmd
end

# Helper to chown files to unprivileged user
function chown_unprivileged(path::AbstractString; uid::Union{Int,Nothing}=get_bash_uid())
    if uid === nothing
        # No UID specified, do nothing
        return false
    end
    
    # Run chown command to change ownership
    # Using just the UID (not changing group)
    try
        run(`chown $uid $path`)
        return true
    catch e
        # Let the error propagate with context
        throw(ErrorException("Failed to change ownership of $path to UID $uid: $e"))
    end
end

end # module