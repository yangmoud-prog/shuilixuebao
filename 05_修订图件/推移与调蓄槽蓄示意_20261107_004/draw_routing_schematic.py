from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea


OUT = Path(__file__).resolve().parent
FONT_DIR = Path("C:/Windows/Fonts")
for name in ("simsun.ttc", "times.ttf", "timesi.ttf", "timesbd.ttf"):
    font_manager.fontManager.addfont(FONT_DIR / name)
SONG = font_manager.FontProperties(family="SimSun", fname=FONT_DIR / "simsun.ttc")
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "mathtext.sf": "Times New Roman",
    "mathtext.cal": "Times New Roman:italic",
    "mathtext.tt": "Times New Roman",
    "mathtext.fallback": None,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.linewidth": 0.9,
    "hatch.linewidth": 0.55,
})

INK = "#26323a"
GRAY = "#696969"
BLUE = "#245d80"
PURPLE = "#65467e"
AMBER = "#93502c"


def pulse(time, duration, rise, fall):
    x = np.clip(time / duration, 0.0, 1.0)
    peak = rise / (rise + fall)
    return x**rise * (1 - x)**fall / (peak**rise * (1 - peak)**fall)


def named_label(ax, x, y, prefix, formula, color, align=0.5, size=11):
    label = HPacker(children=[
        TextArea(prefix, textprops={"fontproperties": SONG, "fontsize": size, "color": color}),
        TextArea(formula, textprops={"fontsize": size, "color": color}),
    ], align="baseline", pad=0, sep=3)
    ax.add_artist(AnnotationBbox(label, (x, y), frameon=False, pad=0,
                                 box_alignment=(align, 0.5), zorder=7))


def main():
    # Analytic, dimensionless hydrographs for a schematic, not experimental data.
    t = np.linspace(0.0, 10.1, 2021)
    tau, sample = 1.75, 3.25
    inflow = pulse(t, 6.0, 2.6, 5.5)
    delayed = pulse(t - tau, 6.0, 2.6, 5.5)
    outflow = pulse(t - tau, 8.15, 3.0, 4.8)
    integrate = getattr(np, "trapezoid", None) or np.trapz
    out_scale = integrate(inflow, t) / integrate(outflow, t)
    outflow *= out_scale
    region = t <= sample + 1e-9
    x = t[region]
    upper_in, shifted_in, outlet = inflow[region], delayed[region], outflow[region]
    lower_in = np.minimum(upper_in, shifted_in)
    assert np.isclose(integrate(inflow, t), integrate(delayed, t))
    assert np.isclose(integrate(delayed, t), integrate(outflow, t))
    assert np.min(np.cumsum(delayed - outflow)) >= -1e-6
    assert np.isclose(x[-1], sample)
    assert shifted_in[-1] > upper_in[-1] > outlet[-1]
    assert np.all(lower_in >= outlet - 1e-12)
    s1 = integrate(upper_in - lower_in, x)
    s2 = integrate(shifted_in - lower_in, x)
    s3 = integrate(lower_in - outlet, x)
    assert min(s1, s2, s3) > 0
    assert np.isclose(s1 - s2, integrate(upper_in - shifted_in, x), atol=1e-12)
    assert np.isclose(s2 + s3, integrate(shifted_in - outlet, x), atol=1e-12)
    assert np.isclose(s1 + s3, integrate(upper_in - outlet, x), atol=1e-12)
    assert np.all(outflow[t <= tau] == 0) and integrate(inflow[t <= tau], t[t <= tau]) > 0

    fig = plt.figure(figsize=(6.18, 3.33), facecolor="white")
    ax = fig.add_axes((0.080, 0.065, 0.895, 0.905))
    ax.set(xlim=(-0.13, 10.23), ylim=(-0.181, 1.21))
    ax.set_axis_off()

    ax.fill_between(x, shifted_in, upper_in, where=upper_in > shifted_in,
                    interpolate=True, facecolor="#e2eef6", edgecolor="#7a9bae",
                    hatch="////", linewidth=0.0, zorder=1)
    ax.fill_between(x, upper_in, shifted_in, where=shifted_in > upper_in,
                    interpolate=True, facecolor="#eee8f2", edgecolor="#947fa6",
                    hatch="---", linewidth=0.0, zorder=1)
    ax.fill_between(x, outlet, lower_in, where=lower_in > outlet,
                    facecolor="#f8eadf", edgecolor="#bd9879",
                    hatch="\\\\", linewidth=0.0, zorder=1)
    ax.vlines(sample, 0, shifted_in[-1], color=GRAY, lw=0.8, zorder=2)
    ax.plot(t[t <= 6.0], inflow[t <= 6.0], color=INK, lw=1.65, zorder=4)
    mid = (t >= tau) & (t <= 6.0 + tau)
    ax.plot(t[mid], delayed[mid], color=BLUE, lw=1.55,
            ls=(0, (4.4, 2.5)), zorder=4)
    tail = (t >= tau) & (t <= 8.15 + tau)
    ax.plot(t[tail], outflow[tail], color=AMBER, lw=1.65, zorder=4)

    ax.annotate("", (10.18, 0), (0, 0),
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 0.95,
                            "shrinkA": 0, "shrinkB": 0, "mutation_scale": 9})
    ax.annotate("", (0, 1.135), (0, 0),
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 0.95,
                            "shrinkA": 0, "shrinkB": 0, "mutation_scale": 9})
    ax.text(-0.026, 0.89, "\u6d41\u91cf", transform=ax.transAxes,
            fontproperties=SONG, fontsize=12, ha="center", va="center",
            rotation=90, rotation_mode="anchor")
    ax.text(10.01, -0.072, "\u65f6\u95f4", fontproperties=SONG,
            fontsize=12, ha="right", va="center")
    ax.text(-0.075, -0.019, "$0$", color=INK, ha="right", va="top")
    ax.text(tau, -0.028, r"$t_0$", color=INK, ha="center", va="top")
    ax.text(sample, -0.025, r"$t$", color=INK, ha="center", va="top")
    named_label(ax, 1.69, 1.095, "\u4e0a\u6e38\u5165\u6d41", r"$I(t)$", INK)
    named_label(ax, 3.98, 1.095, "\u63a8\u79fb\u540e\u5165\u6d41", r"$I(t-\tau)$", BLUE)
    named_label(ax, 5.83, 0.70, "\u4e0b\u6e38\u51fa\u6d41", r"$O(t)$", AMBER, align=0.0)
    area_label = {"facecolor": "white", "edgecolor": "none", "pad": 0.6}
    ax.text(1.43, 0.36, r"$S_1$", color=BLUE, ha="center", va="center", bbox=area_label)
    ax.annotate(r"$S_2$", (3.13, 0.665), (3.50, 0.795), color=PURPLE,
                ha="left", va="center",
                arrowprops={"arrowstyle": "-", "color": PURPLE, "lw": 0.7}, zorder=6)
    ax.text(2.73, 0.265, r"$S_3$", color=AMBER, ha="center", va="center", bbox=area_label)

    ax.vlines((0.0, tau), -0.151, 0.0, color=GRAY, linewidth=0.72, zorder=2)
    ax.annotate("", (tau, -0.145), (0, -0.145), arrowprops={
        "arrowstyle": "<->", "color": BLUE, "lw": 0.85,
        "mutation_scale": 8, "shrinkA": 0, "shrinkB": 0,
    })
    named_label(ax, tau / 2, -0.108, "\u6709\u6548\u4f20\u64ad\u65f6\u95f4", r"$\tau$", BLUE, size=9.0)

    stem = OUT / "\u63a8\u79fb\u4e0e\u8c03\u84c4\u69fd\u84c4\u793a\u610f"
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


if __name__ == "__main__":
    main()
