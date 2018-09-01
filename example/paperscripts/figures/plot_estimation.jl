
using Rsvg, Plots, JLD, LaTeXStrings
plotlyjs()


###
#
# Helper functions:
#
###

function safe_stds(matslice, idx)
  map(x->try sqrt(diag(inv(x))[idx])*1.96 catch 0.0 end, matslice)
end

function plot_slice!(pt::Plots.Plot, idx, pointslice, matslice, sz, step, oset, color; 
                     altlen=nothing, draw_line=false)
  # Find the plot locations:
  if altlen == nothing
    plotspots = collect(linspace(sz+oset-step, sz+oset+step, length(pointslice)))
  else
    plotspots = collect(linspace(sz+oset-step, sz+oset+step, altlen))[1:length(pointslice)]
  end
  # Add the first and second parameter point estimates:
  scatter!(plotspots, map(x->x[idx], pointslice), color=color, markershape=:hexagon, markersize=4,
           markerstrokealpha=0.0)
  if draw_line
    plot!(plotspots, map(x->x[idx], pointslice), color=color, linewidth=0.5)
  end
  # Compute the standard deviations for each:
  stds = safe_stds(matslice, idx)
  # Add them to the plot:
  for j in eachindex(pointslice)
    plot!([plotspots[j], plotspots[j]], [pointslice[j][idx]-stds[j], pointslice[j][idx]+stds[j]],
          color=color, linewidth=1.0, alpha=1.0, label="")
  end
end


###
#
# Main script:
#
###

plotv   = Vector{Vector{Plots.Plot{Plots.PlotlyJSBackend}}}(2)
szs     = log2.([2^11, 2^12, 2^13, 2^14, 2^15, 2^16, 2^17])

for (pind, file) in enumerate(["/path/to/estimates_matern_bigrange.jld",
                               "/path/to/estimates_matern_smallrange.jld"])

  dat   = load(file)
  splot = Vector{Plots.Plot{Plots.PlotlyJSBackend}}(2)
  spts  = collect(linspace(0.5-0.36, 0.5+0.36, 7))
  xtcks = (vcat(map(x->spts .+ x, 0.0:4.0)...), Iterators.repeat(11:17, outer=5))
  for idx in 1:2
    ylm   = ifelse(contains(file, "big"),
                   ifelse(idx==1, (dat["true_parm"][1]-0.3, dat["true_parm"][1]+0.3), (15.0, 130.0)),
                   ifelse(idx==1, (dat["true_parm"][1]-0.4,  dat["true_parm"][1]+0.4), 
                                  (dat["true_parm"][2]-0.75, dat["true_parm"][2]+1.15)))
    ptnew = plot(size=(1250, 625), xlim=(0.0, 5.0), xticks=ifelse(idx==1, nothing, xtcks), leg=false, 
                 ylabel=ifelse(idx==1, "θ₀", "θ₁"), xgrid=false, ylim=ylm, grid=false)
    plot!([dat["true_parm"][idx]], linetype=:hline, color=:black)
    plot!([1.0, 2.0, 3.0, 4.0], linetype=:vline, color=:black, linestyle=:dot)
    locs = collect(linspace(0.1, 0.9, 7))
    eind = collect(1.0:3.0).*(2*0.36/7)
    for j in 1:5
      plot_slice!(ptnew, idx, dat["hodlr_fit"][:,j],   dat["hodlr_hes"][:,j],   j-0.5, 0.36, 0.03,
                  :red, draw_line=true)
      plot_slice!(ptnew, idx, dat["exact_fit"][1:3,j], dat["exact_hes"][1:3,j], j-0.5, 0.36, 0.0,  
                  :blue, altlen=7)
    end
    splot[idx] = ptnew
  end
  plotv[pind] = splot
end