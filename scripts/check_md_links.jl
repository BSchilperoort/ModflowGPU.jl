# Check that every relative Markdown link in docs/dev/, PLAN.md, README.md and CLAUDE.md
# resolves to an existing file (anchors and URLs are skipped).
# Usage: julia scripts/check_md_links.jl
root = dirname(@__DIR__)
files = String[]
for (dir, _, fs) in walkdir(joinpath(root, "docs", "dev"))
    append!(files, joinpath.(dir, filter(endswith(".md"), fs)))
end
append!(files, joinpath.(root, ["PLAN.md", "README.md", "CLAUDE.md"]))
global bad = 0
global nlinks = 0
for f in files
    for (i, line) in enumerate(eachline(f)), m in eachmatch(r"\]\(([^)\s]+)\)", line)
        target = m.captures[1]
        (startswith(target, "http") || startswith(target, "#") || startswith(target, "mailto:")) && continue
        global nlinks += 1
        path = joinpath(dirname(f), first(split(target, '#')))
        if !ispath(path)
            global bad += 1
            println(relpath(f, root), ":", i, ": broken link ", target)
        end
    end
end
println("checked ", nlinks, " relative links in ", length(files), " files, ", bad, " broken")
exit(bad == 0 ? 0 : 1)
