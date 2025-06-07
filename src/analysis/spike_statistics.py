from src.analysis.spike_raster_plot import save_spike_raster_plots, save_spike_raster_plot, save_all_layer_spike_rasters
from src.training.dataset_select import get_dataset_obj
from pathlib import Path
from src.analysis.firing_rate_plot import save_firing_rate_plot
from src.analysis.sparsity_plot import save_sparsity_plot


def generate_spike_statistics_for_model(model, save_dir: Path):
    """
    Loop through all modules, for any module that has `spike_history`,
    generate and save both the firing rate and sparsity plot.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, module in model.named_modules():
        if hasattr(module, "spike_history") and isinstance(module.spike_history, list) and len(module.spike_history) > 0:
            layer_name = name.replace(".", "_")
            save_firing_rate_plot(module.spike_history, save_dir / f"{layer_name}_firing_rate.png")
            save_sparsity_plot(module.spike_history, save_dir / f"{layer_name}_sparsity.png")