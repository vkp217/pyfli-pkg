import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
from scipy.stats import wasserstein_distance, energy_distance, entropy

class Plotter:
    def __init__(self, *args, values=None, style_config=None, source_names=None):
        self.raw_data = args
        self.values = values
        self.source_names = source_names
        self.stats_results = []
        
        if isinstance(style_config, list):
            self.style_colors = style_config
        elif isinstance(style_config, dict):
            self.style_colors = list(style_config.values())
        else:
            self.style_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6']
        
        self.labels = []
        self._clean_data()

    def _get_clean_array(self, data_source, key):
        try:
            val = data_source[key]
            return np.asanyarray(val).astype(float).flatten()
        except (KeyError, ValueError, TypeError):
            return np.array([])

    def _clean_data(self):
        if self.values:
            self.labels = self.values
        else:
            all_keys = []
            for data in self.raw_data:
                if isinstance(data, dict) or str(type(data)) == "<class 'numpy.lib.npyio.NpzFile'>":
                    keys = data.files if hasattr(data, 'files') else data.keys()
                    all_keys.extend(keys)
            self.labels = list(dict.fromkeys(all_keys))

    def make_plot(self, title="Data Analysis", graph_type="box", point_type="strip",
              show_mean=True, show_median=True, y_range=None, 
              show_significance=True, test_type='welch', correction=False):
        """
        graph_type: 'box', 'violin', 'swarm', 'overlay', 'raincloud', 'kde'
        point_type: 'swarm' or 'strip'
        """
        self.stats_results = []
        p_val_text = []
        data_groups = {key: [] for key in self.labels}

        for source in self.raw_data:
            for key in self.labels:
                arr = self._get_clean_array(source, key)
                if arr.size > 0:
                    data_groups[key].append(arr)
        n_sources = len(self.raw_data)
    # KDE PER KEY
        if graph_type == 'kde':
            source_labels = self.source_names or [f"Source {i+1}" for i in range(len(self.raw_data))]
            n_keys = len(self.labels)
            fig, axes = plt.subplots(n_keys, 1, figsize=(10,4*n_keys), sharex=False)
            if n_keys == 1:
                axes = [axes]
            for idx, key in enumerate(self.labels):
                ax = axes[idx]
                for i in range(len(self.raw_data)):
                    if len(data_groups[key]) > i:
                        sns.kdeplot(
                            data_groups[key][i],
                            ax=ax,
                            fill=True,
                            alpha=0.35,
                            color=self.style_colors[i],
                            label=source_labels[i])
                ax.set_title(key)
            plt.suptitle(title)
            src_labels = self.source_names or [f"Source {i+1}" for i in range(n_sources)]
            legend_elements = [
                Line2D(
                    [0], [0],
                    marker='s',
                    color=self.style_colors[i],
                    label=src_labels[i],
                    markersize=10,
                    linestyle='None') for i in range(n_sources)]
            ax.legend(handles=legend_elements, frameon=False, loc='upper left', bbox_to_anchor=(1.02,1))
            plt.tight_layout()
            plt.subplots_adjust(right=0.8)
            self.current_fig = fig
            return

    # LONG DATA FORMAT
        long_data = []

        for i, key in enumerate(self.labels):
            for source_idx in range(n_sources):
                if len(data_groups[key]) > source_idx:
                    data_arr = data_groups[key][source_idx]
                    samp_data = data_arr if len(data_arr) < 250 else np.random.choice(data_arr,250)
                    long_data.extend([{"Key":key,"Source":source_idx,"Value":v} for v in samp_data])
        df = pd.DataFrame(long_data)
        palette = sns.color_palette(self.style_colors[:n_sources])
        fig, ax = plt.subplots(figsize=(14,8))
    # BOX / SWARM / OVERLAY
        if graph_type in ['box','swarm','overlay']:
            width = 0.6 / n_sources
            x_centers = np.arange(len(self.labels))
            for src in range(n_sources):
                color = self.style_colors[src]
                positions = []
                data_list = []
                for idx,key in enumerate(self.labels):
                    if len(data_groups[key]) > src:
                        offset = (src-(n_sources-1)/2) * width*1.2
                        pos = x_centers[idx] + offset
                        positions.append(pos)
                        data_list.append(data_groups[key][src])
            # BOX PLOT
                if graph_type in ['box','overlay']:
                    ax.boxplot(
                        data_list,
                        positions=positions,
                        widths=width*0.9,
                        patch_artist=True,
                        showfliers=False,
                        boxprops=dict(facecolor=color,alpha=0.5),
                        medianprops=dict(color='black'))
            # SWARM / STRIP
                if graph_type in ['swarm','overlay']:
                    for pos,data_arr in zip(positions,data_list):
                        if point_type == "swarm":
                            sns.swarmplot(
                            x=np.repeat(pos,len(data_arr)),
                            y=data_arr,
                            ax=ax,
                            color=color,
                            size=4,
                            edgecolor="black",
                            linewidth=0.4)                  
                        else:
                            ax.scatter(np.random.normal(pos,width*0.08,len(data_arr)),
                                data_arr,
                                s=12,
                                color=color,
                                alpha=0.6)
            # MEAN / MEDIAN
                for pos,data_arr in zip(positions,data_list):
                    if show_mean:
                        ax.scatter(
                            pos,
                            np.nanmean(data_arr),
                            color='white',
                            edgecolor='black',
                            s=40,
                            zorder=5)                
                    if show_median:
                        ax.hlines(
                            np.nanmedian(data_arr),
                            pos-width/3,
                            pos+width/3,
                            color='black',
                            lw=2)
            ax.set_xticks(x_centers)
            ax.set_xticklabels(self.labels)

    # VIOLIN / RAINCLOUD
        elif graph_type in ['violin','raincloud']:
            x_centers = np.arange(len(self.labels))
            width = 0.6 / n_sources
            for src in range(n_sources):
                color = self.style_colors[src]
                for idx,key in enumerate(self.labels):
                    if len(data_groups[key]) > src:
                        data_arr = data_groups[key][src]
                        offset = (src-(n_sources-1)/2)*width*1.2
                        pos = x_centers[idx] + offset
                    # FULL VIOLIN
                        if graph_type == "violin":
                            v = ax.violinplot(
                                data_arr,
                                positions=[pos],
                                widths=width,
                                showmeans=False,
                                showmedians=False,
                                showextrema=True)                     
                            for partname in ('cbars','cmins','cmaxes','cmedians'):
                                if partname in v:
                                    v[partname].set_edgecolor('black')
                                    v[partname].set_linewidth(1.2)
                            for body in v['bodies']:
                                body.set_facecolor(color)
                                body.set_alpha(0.6)
                                body.set_edgecolor("black")
                                body.set_linewidth(1.2)
                    # RAINCLOUD (HALF VIOLIN)
                        if graph_type == "raincloud":
                            v = ax.violinplot(
                                data_arr,
                                positions=[pos],
                                widths=width,
                                showextrema=False)                    
                            body = v['bodies'][0]
                            path = body.get_paths()[0]
                            vertices = path.vertices
                            mean_x = np.mean(vertices[:,0])
                        # clip left side
                            vertices[:,0] = np.clip(vertices[:,0], mean_x, np.inf)
                            body.set_facecolor(color)
                            body.set_alpha(0.4)
                            body.set_edgecolor("black")
                            # BOX overlay
                            ax.boxplot(data_arr,
                                positions=[pos],
                                widths=width*0.5,
                                patch_artist=True,
                                showfliers=False,
                                boxprops=dict(facecolor=color,alpha=0.7),
                                medianprops=dict(color='black'))
                    # Mean / Median markers
                        if show_mean:
                            ax.scatter(pos,
                                np.nanmean(data_arr),
                                color='white',
                                edgecolor='black',
                                s=40,
                                zorder=6)
                        if show_median:
                            median_val = np.nanmedian(data_arr)
                            ax.hlines(
                                np.nanmedian(data_arr),
                                pos-width/3,
                                pos+width/3,
                                color='black',
                                lw=2,
                                zorder=6)
            ax.set_xticks(np.arange(len(self.labels)))
            ax.set_xticklabels(self.labels)
    # SIGNIFICANCE TESTING
        if show_significance and test_type.lower() != 'none' and n_sources >= 2:
            p_val_text = []
            num_comps = len(self.labels) * (n_sources-1) if correction else 1
            for idx,key in enumerate(self.labels):
                if len(data_groups[key]) >= 2:
                    s1 = data_groups[key][0]
                    y_max = max([np.nanmax(d) for d in data_groups[key]])
                    for src in range(1,n_sources):
                        s2 = data_groups[key][src]
                        if test_type.lower() == "paired":
                            m = min(len(s1),len(s2))
                            _,p = stats.ttest_rel(s1[:m],s2[:m])
                        else:
                            _,p = stats.ttest_ind(s1,s2,equal_var=False)
                        adj_p = min(1.0,p*num_comps)
                        if adj_p < 0.001:
                            sig="***"
                        elif adj_p < 0.01:
                            sig="**"
                        elif adj_p < 0.05:
                            sig="*"
                        else:
                            sig="NS"
                        offset = (src-(n_sources-1)/2)*(0.6/n_sources)*1.2
                        ymin, ymax = ax.get_ylim()
                        y_range1 = ymax - ymin
                        star_height = ymax - 0.05*y_range1
                        ax.text(
                            idx + offset,
                            star_height,
                            sig,
                            ha='center',
                            fontsize=11,
                            fontweight='bold')
                        p_val_text.append(f"{key} vs S{src+1}: {adj_p:.2e}")
                        self.stats_results.append({
                            "Key":key,
                            "Source":src+1,
                            "P":adj_p
                        })
        if p_val_text:
            ax.text(1.02,
                0.4,
                "P-Values:\n" + "\n".join(p_val_text),
                transform=ax.transAxes,
                fontsize=8,
                bbox=dict(boxstyle='round', facecolor='none', edgecolor='none', alpha=0.4))
    # LEGEND (for non-KDE plots)
        src_labels = self.source_names or [f"Source {i+1}" for i in range(n_sources)]
        legend_elements = [Line2D(
                [0], [0],
                marker='s',
                color=self.style_colors[i],
                label=src_labels[i],
                markersize=10,
                linestyle='None'
            )
            for i in range(n_sources)]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.tight_layout()
        self.current_fig = fig
    def export_data(self, save_pdf=False, save_csv=False, filename="results"):
        if save_pdf and hasattr(self,'current_fig'):
            self.current_fig.savefig( f"{filename}.pdf", format='pdf', bbox_inches='tight')
        if save_csv and self.stats_results:
            pd.DataFrame(self.stats_results).to_csv(f"{filename}.csv", index=False)

####### Specifically for DL model and comparison of the output values
class DLModelComparator(Plotter):
    def compute_distribution_metrics(self):
        results = []
        gt_index = 0  # assume first dataset is ground truth
        for key in self.labels:
            gt = self._get_clean_array(self.raw_data[gt_index], key)
            for i in range(1, len(self.raw_data)):
                model = self._get_clean_array(self.raw_data[i], key)
                min_len = min(len(gt), len(model))
                gt_s = gt[:min_len]
                model_s = model[:min_len]
                # Wasserstein
                w = wasserstein_distance(gt_s, model_s)
                # Energy
                e = energy_distance(gt_s, model_s)
                # KL divergence
                hist_gt, bins = np.histogram(gt_s, bins=50, density=True)
                hist_model, _ = np.histogram(model_s, bins=bins, density=True)
                hist_gt += 1e-10
                hist_model += 1e-10
                kl = entropy(hist_gt, hist_model)
                results.append({
                    "Key": key,
                    "Model": i,
                    "Wasserstein": w,
                    "Energy": e,
                    "KL": kl
                })
        return results
    
    def annotate_distribution_metrics(self, ax):
        metrics = self.compute_distribution_metrics()
        lines = []
        for m in metrics:
            text = (f"{m['Key']} / M{m['Model']}: "
                f"W={m['Wasserstein']:.3f}, "
                f"E={m['Energy']:.3f}, "
                f"KL={m['KL']:.3f}")                
            lines.append(text)
        ax.text(
            1.02,
            0.12,
            "Distribution Metrics\n" + "\n".join(lines),
            transform=ax.transAxes,
            fontsize=8)