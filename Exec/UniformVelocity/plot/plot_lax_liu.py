import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys, glob, argparse
from plot_utils import find_final_plotfile, read_amrex_plotfile

RESOLUTIONS = [320, 640, 1280]

def plot_single(ax, x, y, density, title, vmin=None, vmax=None):
    im = ax.imshow(density.T, origin='lower',
                   extent=[x[0], x[-1], y[0], y[-1]],
                   cmap='jet', vmin=vmin, vmax=vmax,
                   interpolation='bilinear', aspect='equal')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return im

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res', type=int, default=None)
    parser.add_argument('--outdir', type=str, default='plot/result/result_lax_liu')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    resolutions = [args.res] if args.res else RESOLUTIONS

    datasets = []
    for res in resolutions:
        base = f'output_lax_liu/no_amr/lax_liu_{res}'
        try:
            pf = find_final_plotfile(base)
            print(f"Loading {res}x{res}: {pf}")
            x, y, data, var_names, time = read_amrex_plotfile(pf)
            density = data[:, :, 0]
            datasets.append((res, x, y, density, time))
        except FileNotFoundError as e:
            print(f"WARNING: {e} — skipping {res}")

    if not datasets:
        print("No plotfiles found.")
        return

    vmin = min(d[3].min() for d in datasets)
    vmax = max(d[3].max() for d in datasets)

    if len(datasets) == 1:
        res, x, y, density, time = datasets[0]
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        im = plot_single(ax, x, y, density,
                         f'Lax-Liu Config 3 — {res}x{res}  (t={time:.3f})',
                         vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label='Density')
        plt.tight_layout()
        out = os.path.join(args.outdir, f'lax_liu_{res}.png')
        plt.savefig(out, dpi=150)
        print(f"Saved: {out}")
        return

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, len(datasets) + 1,
                           width_ratios=[1] * len(datasets) + [0.05],
                           wspace=0.3)
    axes = [fig.add_subplot(gs[i]) for i in range(len(datasets))]
    cax  = fig.add_subplot(gs[-1])

    im_last = None
    for ax, (res, x, y, density, time) in zip(axes, datasets):
        im = plot_single(ax, x, y, density, f'{res}x{res}', vmin=vmin, vmax=vmax)
        im_last = im

    fig.colorbar(im_last, cax=cax, label='Density')
    fig.suptitle(f'Lax-Liu Configuration 3  (t = {datasets[0][4]:.3f})',
                 fontsize=13, y=1.01)

    out = os.path.join(args.outdir, 'lax_liu_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")

if __name__ == '__main__':
    main()