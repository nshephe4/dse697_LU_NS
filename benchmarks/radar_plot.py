import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib.path import Path
from matplotlib.projections import register_projection

def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` Axes.
    This function creates a RadarAxes projection and registers it.
    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                        radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                spine = Spine(axes=self,
                                spine_type='circle',
                                path=Path.unit_regular_polygon(num_vars))
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5) + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

if __name__ == '__main__':
    # Load and normalize data
    df = pd.read_csv('evaluation_results.csv')
    df['Value'] = df['Value'] / df['Value'].max()

    metrics = df['Metric'].tolist()
    values = df['Value'].tolist()

    N = len(metrics)
    theta = radar_factory(N, frame='polygon')

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='radar'))
    ax.set_title("Model Evaluation Radar Plot", weight='bold', size='medium')

    # Dynamic r-grid scaling
    max_value = max(values)
    padded_max = round(max_value + 0.2, 1)
    num_levels = 5
    grid_levels = np.linspace(0, padded_max, num_levels)
    ax.set_rgrids(grid_levels, labels=[f"{vl:.1f}" for vl in grid_levels])
    ax.set_ylim(0, padded_max)

    # Plot data
    ax.plot(theta, values, 'o-', linewidth=2)
    ax.fill(theta, values, alpha=0.25)
    ax.set_varlabels(metrics)

    for txt in ax.texts:
        txt.set_fontsize(12)

    # Save the plot
    plt.savefig("model_evaluation_radar_plot.png", bbox_inches="tight", dpi=300, format="png")
    plt.close()